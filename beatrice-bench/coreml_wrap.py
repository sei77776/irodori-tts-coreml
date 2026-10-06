"""nn.Module wrapper of the neural part of one pseudo-streaming call, for tracing/Core ML.

Inputs (fixed shapes):
  wav       [1, 1, W*160]  16 kHz, W frames = context + chunk + lookahead
  spk_embed [1, 256]       embed_speaker[spk] + embed_formant_shift[formant_idx]
  kv        [1, 384, 128]  key_value_speaker_embedding[spk]
  codebook  [1, 512, 128]  vq.codebooks[spk]
  pitch_shift_bins [1]     float, semitone * 8 (rounded)
Outputs (per frame, length W): ir_amp [1,257,W], ir_phase [1,257,W],
  aperiodicity [1,240,W], post_filter [1,512,W], pitch_hz [1,W]

`variant` controls op substitutions used to get past converter failures:
  "orig": exactly the trainer's modules
  "safe": FFT -> matmul DFT, MultiheadAttention -> explicit SDPA, VQ via one-hot matmul,
          integer reflect pads -> float
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bt_infer import bt, vocoder_nn


def dft_mats(n, n_bins):
    k = torch.arange(n_bins, dtype=torch.float64)[None, :]
    t = torch.arange(n, dtype=torch.float64)[:, None]
    ang = -2 * math.pi * t * k / n
    return torch.cos(ang).float(), torch.sin(ang).float()  # [n, n_bins]


class PitchFeaturesSafe(nn.Module):
    """extract_pitch_features without complex tensors / FFT (real matmul DFT + direct autocorrelation)."""
    def __init__(self, hop=160, win=560, max_corr=256, corr_win=304, cutoff=64):
        super().__init__()
        self.hop, self.win, self.max_corr, self.corr_win, self.cutoff = hop, win, max_corr, corr_win, cutoff
        c, s = dft_mats(win, cutoff)
        self.register_buffer("dft_c", c)
        self.register_buffer("dft_s", s)
        self.register_buffer("cosw", torch.signal.windows.cosine(win))
        # framing as a strided conv with an identity kernel (aten::unfold is not supported by coremltools)
        self.register_buffer("frame_kernel", torch.eye(win)[:, None, :])

    def forward(self, y):  # y: [B, T]
        pad = (self.win - self.hop) // 2
        y = F.pad(y, (pad, pad))
        n_frames = (y.size(-1) - self.win) // self.hop + 1
        # frames via conv1d with identity kernel (avoids unfold)
        fr = F.conv1d(y[:, None, :], self.frame_kernel, stride=self.hop).transpose(1, 2)  # [B, F, win]
        re = fr @ self.dft_c  # [B, F, 64]
        im = fr @ self.dft_s
        mag = torch.sqrt(re * re + im * im)
        logp = torch.log10(mag + 1e-5)
        # delta = spec[t] * conj(spec[t-1]) normalized
        re0, im0 = re[:, :-1], im[:, :-1]
        re1, im1 = re[:, 1:], im[:, 1:]
        dre = re1 * re0 + im1 * im0
        dim_ = im1 * re0 - re1 * im0
        dn = torch.sqrt(dre * dre + dim_ * dim_) + 1e-5
        dre = F.pad(dre / dn, (0, 0, 1, 0))
        dim_ = F.pad(dim_ / dn, (0, 0, 1, 0))
        inst = torch.cat([logp, dre, dim_], -1).transpose(1, 2)  # [B, 192, F]
        # difference function: for lag tau in [1, max_corr], over the frame's last corr_win samples
        # trainer: flipped frames; corr[k] for k in [corr_win, win) of irfft(rfft(flip(y)) * rfft(y[-corr_win:]))
        flipped = fr.flip(-1)
        seg = fr[..., -self.corr_win:]  # [B,F,304]
        # corr[j] = sum_m flipped[(j - m) mod win] * seg[m], j in [corr_win, win)  -> no wrap since j-m in [0, win)
        # build index matrix once
        j = torch.arange(self.corr_win, self.win)[:, None]
        m = torch.arange(self.corr_win)[None, :]
        idx = (j - m)  # [256, 304] in [1, 559]
        # gather flipped values: [B,F,256,304]
        g = flipped[..., idx.reshape(-1)].reshape(fr.size(0), fr.size(1), self.max_corr, self.corr_win)
        corr = (g * seg[..., None, :]).sum(-1)  # [B,F,256]
        energy = torch.cumsum(flipped * flipped, -1)
        e0 = energy[..., self.corr_win - 1:self.corr_win]
        e = energy[..., self.corr_win:] - energy[..., :-self.corr_win]
        cd = torch.clamp(e0 + e - 2.0 * corr, min=0.0) * (2.0 / self.corr_win)
        cd = torch.sqrt(cd).transpose(1, 2)
        en = ((fr * self.cosw) ** 2).sum(-1, keepdim=True).transpose(1, 2)
        en = torch.log10(torch.clamp(en, min=1e-3)) * 0.5
        return inst, cd, en


class StreamNN(nn.Module):
    def __init__(self, pe, ps, g, variant="orig"):
        super().__init__()
        self.pe, self.ps, self.g, self.variant = pe, ps, g, variant
        if variant == "safe":
            self.pf = PitchFeaturesSafe()
        self.register_buffer("pitch_ids", torch.arange(448, dtype=torch.float32))

    # --- phone extractor -------------------------------------------------
    def phone(self, x):
        if self.variant == "orig":
            return self.pe(x, return_stats=False)
        pe = self.pe
        h = pe.feature_extractor(x)
        h = pe.feature_projection(h)
        bb = pe.backbone
        h = bb.embed(h)
        h = bb.norm(h.transpose(1, 2)).transpose(1, 2)
        B, C, L = h.shape  # L multiple of 4 is required (fixed W chosen accordingly)
        L4 = L // 4
        mask = (torch.ones(L4, L4).triu(1) * -1e4).to(h.dtype)
        for blk in bb.convnext:
            idt = h
            y = h.view(B, C, L4, 4).permute(0, 3, 2, 1).reshape(B * 4, L4, C)
            y = blk.attn_norm(y)
            mha = blk.mha
            qkv = F.linear(y, mha.in_proj_weight, mha.in_proj_bias)
            q, k, v = qkv.chunk(3, -1)
            nh = mha.num_heads; hd = C // nh
            q = q.view(B * 4, L4, nh, hd).transpose(1, 2)
            k = k.view(B * 4, L4, nh, hd).transpose(1, 2)
            v = v.view(B * 4, L4, nh, hd).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(hd)) + mask
            o = att.softmax(-1) @ v
            o = o.transpose(1, 2).reshape(B * 4, L4, C)
            o = mha.out_proj(o)
            o = o.view(B, 4, L4, C).permute(0, 3, 2, 1).reshape(B, C, L)
            h = idt + o
            # conv part of ConvNeXtBlock
            idt = h
            y = blk.dwconv(h).transpose(1, 2)
            y = blk.norm(y)
            y = blk.pwconv2(F.gelu(blk.pwconv1(y), approximate="tanh"))
            y = y * blk.gamma
            h = idt + y.transpose(1, 2)
        h = bb.final_layer_norm(h.transpose(1, 2)).transpose(1, 2)
        return pe.head(F.gelu(h, approximate="tanh"))

    def vq(self, x, codebook):
        q = x / torch.clamp(torch.sqrt((x * x).sum(1, keepdim=True)), min=1e-6)
        sim = torch.einsum("bcl,bkc->blk", q, codebook)  # [1,L,512]
        _, idx = sim.topk(4, dim=-1)
        if self.variant == "orig":
            L = x.size(2)
            gathered = codebook[:, None].expand(-1, L, -1, -1).gather(2, idx[..., None].expand(-1, -1, -1, 128)).mean(2)
            return gathered.transpose(1, 2)
        onehot = F.one_hot(idx, 512).to(x.dtype).sum(2) * 0.25  # [1,L,512]
        return (onehot @ codebook).transpose(1, 2)

    # --- pitch ------------------------------------------------------------
    def pitch(self, x):
        ps = self.ps
        if self.variant == "orig":
            return ps(x)
        inst, cd, en = self.pf(x.squeeze(1))
        a = ps.instfreq_embed_1(F.gelu(ps.instfreq_embed_0(inst), approximate="tanh"))
        b = ps.corr_embed_1(F.gelu(ps.corr_embed_0(cd), approximate="tanh"))
        h = ps.backbone(F.gelu(a + b, approximate="tanh"))
        return ps.head(h), en

    def sample_pitch(self, logits):
        # float-only re-implementation of PitchEstimator.sample_pitch(return_features=True):
        # gathers are written as one-hot masks (argmax indices are compared against an arange)
        p = logits.softmax(1)  # [1,448,L]
        unv = p[:, :1]
        p2 = torch.cat([torch.full_like(p[:, :1], -100.0), p[:, 1:]], 1)
        band = p2[:, :-3] + p2[:, 1:-2] + p2[:, 2:-1] + p2[:, 3:]  # [1,445,L]
        ids = self.pitch_ids[None, :, None]  # [1,448,1]
        bids = ids[:, :445]
        qb = band.argmax(1, keepdim=True).to(self.pitch_ids.dtype)  # [1,1,L]
        pick = lambda t, i: (t * (bids == i).to(self.pitch_ids.dtype)).sum(1, keepdim=True)
        bp = pick(band, qb)
        half = pick(band, torch.clamp(qb - 96.0, min=1.0)) * (qb > 96.0).to(self.pitch_ids.dtype)
        dbl = pick(band, torch.clamp(qb + 96.0, max=444.0)) * (qb <= 444.0 - 96.0).to(self.pitch_ids.dtype)
        mask = ((ids >= qb) & (ids < qb + 4.0)).to(self.pitch_ids.dtype)
        qp = (p2 * mask).argmax(1).to(self.pitch_ids.dtype)  # [1,L]
        feats = torch.cat([unv, half / (bp + 1e-6), dbl / (bp + 1e-6)], 1)
        return qp, feats

    def forward(self, wav, spk_embed, kv, codebook, pitch_shift_bins):
        g = self.g
        phone = self.vq(self.phone(wav), codebook)
        phone = phone * torch.rsqrt((phone * phone).mean(1, keepdim=True) + 1.1920928955078125e-07)
        logits, energy = self.pitch(wav)
        if self.variant == "orig":
            qp, pf = self.ps.sample_pitch(logits, return_features=True)
        else:
            qp, pf = self.sample_pitch(logits)
        qpf = qp.to(self.pitch_ids.dtype) if self.variant == "orig" else qp
        qpf = torch.where(qpf == 0, qpf, torch.clamp(qpf + pitch_shift_bins, 1.0, 447.0))
        pitch_hz = 55.0 * torch.pow(2.0, qpf / 96.0)
        # reflect pads (left by 1 / 2 frames) written as concatenation
        energy = torch.cat([energy[:, :, 1:2], energy[:, :, :-1]], 2)
        qpf_d = torch.cat([qpf[:, 2:3], qpf[:, 1:2], qpf[:, :-2]], 1)
        pf = torch.cat([pf[:, :, 2:3], pf[:, :, 1:2], pf[:, :, :-2]], 2)
        pf = torch.cat([energy, pf], 1)
        if self.variant == "orig":
            emb_p = g.embed_quantized_pitch(qpf_d.long())
        else:
            emb_p = (qpf_d[:, :, None] == self.pitch_ids[None, None, :]).to(self.pitch_ids.dtype) @ g.embed_quantized_pitch.weight
        h = g.embed_phone(phone) + emb_p.transpose(1, 2) + g.embed_pitch_features(pf) + spk_embed[:, :, None]
        h = F.silu(h)
        o = vocoder_nn(g.vocoder, h, kv)
        return o["ir_amp"], o["ir_phase"], o["aperiodicity"], o["post_filter"], pitch_hz


def speaker_inputs(g, spk, formant=0.0):
    fsi = int(round((formant + 2.0) * 2.0))
    se = (g.embed_speaker.weight[spk] + g.embed_formant_shift.weight[fsi])[None]
    kv = g.key_value_speaker_embedding.weight[spk].view(1, 384, 128)
    cb = g.vq.codebooks[spk].float()[None]
    return se, kv, cb
