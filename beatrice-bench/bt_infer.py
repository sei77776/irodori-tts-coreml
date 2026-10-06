"""Beatrice v2 (beatrice-trainer 2.0.0-rc.0) inference helpers for the PoC.

- Loads model classes from beatrice_trainer/__main__.py without running training code
  (pyworld / tensorboard are stubbed; torchaudio.list_audio_backends assertion removed).
- Splits inference into:
    nn_part():  16 kHz wav -> per-frame vocoder parameters (all neural nets)
    dsp_part(): vocoder parameters -> 24 kHz wav (pulse overlap-add, noise, post filter)
  so that the random parts (initial pulse phase, noise excitation) can be made
  deterministic / stateful for streaming comparisons.
"""
import math
import os
import sys
import types
import warnings

import torch
import torch.nn.functional as F

# Work dir holding the beatrice-trainer clone (set by run.sh); defaults to ./work next to this file.
POC = os.environ.get("BEATRICE_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
REPO = os.path.join(POC, "beatrice-trainer")
MAIN = os.path.join(REPO, "beatrice_trainer", "__main__.py")

warnings.filterwarnings("ignore")


def load_bt():
    sys.modules.setdefault("pyworld", types.ModuleType("pyworld"))
    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = tb
    src = open(MAIN, encoding="utf-8").read()
    src = src.replace(
        'assert "soundfile" in torchaudio.list_audio_backends()', "pass", 1
    )
    mod = types.ModuleType("bt")
    mod.__file__ = MAIN
    exec(compile(src, MAIN, "exec"), mod.__dict__)
    return mod


bt = load_bt()


def load_models(device="cpu", deploy=None):
    if deploy is None:
        deploy = os.environ.get("DEPLOY", "1") == "1"
    pa = os.path.join(REPO, "assets", "pretrained")
    pe = bt.PhoneExtractor().eval().requires_grad_(False)
    ck = torch.load(os.path.join(pa, "122_checkpoint_03000000.pt"), map_location="cpu", weights_only=True)
    print("phone_extractor:", pe.load_state_dict(ck["phone_extractor"], strict=False))
    ps = bt.PitchEstimator().eval().requires_grad_(False)
    ck = torch.load(os.path.join(pa, "104_3_checkpoint_00300000.pt"), map_location="cpu", weights_only=True)
    print("pitch_estimator:", ps.load_state_dict(ck["pitch_estimator"]))
    import gzip
    with gzip.open(os.path.join(pa, "151_checkpoint_libritts_r_200_02750000.pt.gz"), "rb") as f:
        ck = torch.load(f, map_location="cpu", weights_only=True)
    ck = {"net_g": ck["net_g"]}
    n_spk = ck["net_g"]["embed_speaker.weight"].size(0)
    g = bt.ConverterNetwork(pe, ps, n_spk, 448, 256).eval().requires_grad_(False)
    print("net_g:", g.load_state_dict(ck["net_g"], strict=False))
    # CrossAttention passes dropout_p=0.1 to F.scaled_dot_product_attention even in eval
    # mode (no `self.training` check), so inference would be stochastic. Disable it.
    for m in g.modules():
        if isinstance(m, bt.CrossAttention):
            m.dropout = 0.0
    if deploy:
        # same folding the trainer does before its fp16 export, except PhoneExtractor's
        # feature_projection->embed fold (not exact with zero padding: ~28 dB SNR, propagated by attention)
        pe.remove_weight_norm(); pe.backbone.merge_weights(); ps.merge_weights(); g.merge_weights()
        # merge_weights() is written for the binary dump: WS layers still re-standardize
        # in forward(), so switch them to plain conv/linear on the merged weights.
        for m in list(pe.modules()) + list(ps.modules()) + list(g.modules()):
            if isinstance(m, bt.WSConv1d):
                m.forward = types.MethodType(_plain_conv_forward, m)
            elif isinstance(m, bt.WSLinear):
                m.forward = types.MethodType(lambda self, inp: F.linear(inp, self.weight, self.bias), m)
        for m in (pe, ps, g):
            m.requires_grad_(False)  # remove_weight_norm() re-creates Parameters with requires_grad=True
    return pe, ps, g


def _plain_conv_forward(self, inp):
    r = F.conv1d(inp, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return r if self.trim == 0 else r[:, :, : -self.trim]


@torch.inference_mode()
def nn_part(pe, ps, g, x, spk: int, formant: float = 0.0, pitch_shift: float = 0.0):
    """x: [1,1,T16] (T16 % 160 == 0). Returns dict of per-frame tensors (length = T16/160)."""
    spk_t = torch.tensor([spk])
    # phone extractor + VQ (VQ is applied as a forward hook on pe.head in the trainer)
    phone = pe(x, return_stats=False)  # [1,128,L]
    phone = g.vq(phone, spk_t)
    phone = phone * (1.0 / phone.square().mean(1, keepdim=True).add(torch.finfo(torch.float).eps).sqrt())
    pitch_logits, energy = ps(x)
    qp, pf = ps.sample_pitch(pitch_logits, return_features=True)
    if pitch_shift:
        qp = torch.where(qp == 0, qp, (qp + round(pitch_shift * 96 / 12.0)).clamp(1, 447))
    pitch = 55.0 * 2.0 ** (qp.float() / 96)
    energy = F.pad(energy[:, :, :-1], (1, 0), mode="reflect")
    qp = F.pad(qp[:, :-2], (2, 0), mode="reflect")
    pf = F.pad(pf[:, :, :-2], (2, 0), mode="reflect")
    pf = torch.cat([energy, pf], 1)
    fsi = torch.tensor([int(round((formant + 2.0) * 2.0))])
    h = (g.embed_phone(phone) + g.embed_quantized_pitch(qp).transpose(1, 2)
         + g.embed_pitch_features(pf)
         + (g.embed_speaker(spk_t)[:, :, None] + g.embed_formant_shift(fsi)[:, :, None]))
    h = F.silu(h)
    kv = g.key_value_speaker_embedding(spk_t).view(1, 384, 128)
    return vocoder_nn(g.vocoder, h, kv) | {"pitch": pitch}


def vocoder_nn(v, h, kv):
    x = v.prenet(h, kv)
    ir = F.silu(v.ir_generator(x))
    ir = v.ir_generator_post(ir) * v.ir_scale
    ir_amp = ir[:, :257, :].exp()
    ir_phase = F.pad(ir[:, 257:, :], (0, 0, 1, 1))
    ir_phase = ir_phase.clone()
    ir_phase[:, 1::2, :] += math.pi
    ap = F.silu(v.aperiodicity_generator(x))
    ap = v.aperiodicity_generator_post(ap) * v.aperiodicity_scale
    pf = F.silu(v.post_filter_generator(x))
    pf = v.post_filter_generator_post(pf) * v.post_filter_scale
    pf = pf.clone()
    pf[:, 0, :] += 1.0
    return {"ir_amp": ir_amp, "ir_phase": ir_phase, "aperiodicity": ap, "post_filter": pf}


def overlap_add_det(ir_amp, ir_phase, window, pitch, hop=240, sr=24000.0, init_phase=0.0):
    """bt.overlap_add with a given initial phase instead of torch.rand. Returns (signal, final_phase)."""
    B, irl, L = ir_amp.size()
    irl = (irl - 1) * 2
    nf = pitch / sr
    nf = nf.clone()
    nf[:, 0] = init_phase
    cum = nf.double().cumsum(1)
    phase = (cum % 1.0).float()
    i0, i1 = torch.nonzero(phase[:, :-1] > phase[:, 1:], as_tuple=True)
    numer = 1.0 - phase[i0, i1]
    frac = numer / (numer + phase[i0, i1 + 1])
    a = ir_amp[i0, :, i1 // hop]
    p = ir_phase[i0, :, i1 // hop]
    dp = torch.arange(irl // 2 + 1, dtype=torch.float32)[None, :] * (-math.tau / irl) * frac[:, None]
    ir = torch.fft.irfft(torch.polar(a, p + dp), n=irl, dim=1) * window
    ir = ir.ravel()
    i0e = i0[:, None].expand(-1, irl).ravel()
    i1e = (i1[:, None] + torch.arange(irl)).ravel()
    out = torch.zeros((B, L * hop + irl))
    out.index_put_((i0e, i1e), ir, accumulate=True)
    return out[:, : -irl], cum


def noise_from_excitation(ap, excitation, hop=240):
    """bt.generate_noise with a given excitation of length (L+1)*hop."""
    B, _, L = ap.size()
    n_fft = 2 * hop
    noise = torch.stft(excitation, n_fft=n_fft, hop_length=hop, window=torch.ones(n_fft),
                       center=False, return_complex=True)
    noise[:, 0, :] = 0.0
    noise[:, 1:, :] *= ap
    noise = torch.fft.irfft(noise, n=n_fft, dim=1) * torch.hann_window(n_fft)[None, :, None]
    noise = F.fold(noise, (1, (L + 1) * hop), (1, n_fft), stride=(1, hop)).squeeze(1).squeeze(1)
    return noise[:, : -hop]


@torch.inference_mode()
def dsp_part(v, p, init_phase, excitation, hop=240):
    """p: dict from nn_part. excitation: [1,(L+1)*hop] uniform noise in [-0.5,0.5).
    Returns (wav, periodic, aperiodic, cumulative phase per sample [1, L*hop] float64)."""
    L = p["ir_amp"].size(2)
    pitch = torch.repeat_interleave(p["pitch"], hop, dim=1)
    per, cum = overlap_add_det(p["ir_amp"], p["ir_phase"], v.ir_window, pitch, hop, 24000.0, init_phase)
    aper = noise_from_excitation(p["aperiodicity"], excitation, hop)
    pf = torch.fft.rfft(p["post_filter"].transpose(1, 2), n=768)
    def filt(s):
        s = torch.fft.irfft(torch.fft.rfft(s.view(1, L, hop), n=768) * pf, n=768)
        return F.fold(s.transpose(1, 2), (1, (L - 1) * hop + 768), (1, 768), stride=(1, hop)).squeeze(1).squeeze(1)
    per = filt(per)[:, 120: 120 + L * hop]
    aper = filt(aper)[:, 120: 120 + L * hop]
    return per + aper, per, aper, cum
