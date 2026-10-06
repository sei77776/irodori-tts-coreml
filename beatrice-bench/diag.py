"""Stage-by-stage diagnosis of Core ML vs PyTorch mismatch (run on macOS).

1. DebugNN = StreamNN('safe') that also returns every intermediate tensor.
   Converted to FP32 and FP16, run with CPU_ONLY, compared to PyTorch eager FP32 per stage.
2. Single-op unit models (topk / argmax / one_hot / cumsum / gather / softmax-argmax / pow)
   in FP32 and FP16, compared to numpy/PyTorch.
Writes diag.json / diag.md into the results dir.
"""
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_infer import REPO, vocoder_nn  # noqa: E402
from coreml_wrap import StreamNN  # noqa: E402

import coremltools as ct  # noqa: E402

STAGES = ["inst", "corr", "energy_raw", "phone_raw", "phone_vq", "phone_norm", "pitch_logits", "pitch_softmax",
          "qp", "pitch_feats", "qpf", "expo", "pow_raw", "pitch_hz", "h", "prenet", "ir_amp", "ir_phase", "aperiodicity", "post_filter"]
INT_STAGES = {"qp", "qpf"}


class DebugNN(StreamNN):
    def forward(self, wav, spk_embed, kv, codebook, pitch_shift_bins):
        g = self.g
        inst, cd, en = self.pf(wav.squeeze(1))
        phone_raw = self.phone(wav)
        phone_vq = self.vq(phone_raw, codebook)
        phone = phone_vq * torch.rsqrt((phone_vq * phone_vq).mean(1, keepdim=True) + 1.1920928955078125e-07)
        ps = self.ps
        a = ps.instfreq_embed_1(F.gelu(ps.instfreq_embed_0(inst), approximate="tanh"))
        b = ps.corr_embed_1(F.gelu(ps.corr_embed_0(cd), approximate="tanh"))
        logits = ps.head(ps.backbone(F.gelu(a + b, approximate="tanh")))
        qp, pf = self.sample_pitch(logits)
        qpf = torch.where(qp == 0, qp, torch.clamp(qp + pitch_shift_bins, 1.0, 447.0))
        expo = qpf / 96.0
        pow_raw = torch.pow(2.0, expo)
        pitch_hz = 55.0 * pow_raw
        energy = torch.cat([en[:, :, 1:2], en[:, :, :-1]], 2)
        qpf_d = torch.cat([qpf[:, 2:3], qpf[:, 1:2], qpf[:, :-2]], 1)
        pf2 = torch.cat([pf[:, :, 2:3], pf[:, :, 1:2], pf[:, :, :-2]], 2)
        pf2 = torch.cat([energy, pf2], 1)
        emb_p = (qpf_d[:, :, None] == self.pitch_ids[None, None, :]).to(self.pitch_ids.dtype) @ g.embed_quantized_pitch.weight
        h = g.embed_phone(phone) + emb_p.transpose(1, 2) + g.embed_pitch_features(pf2) + spk_embed[:, :, None]
        h = F.silu(h)
        x = g.vocoder.prenet(h, kv)
        o = vocoder_nn(g.vocoder, h, kv)
        return (inst, cd, en, phone_raw, phone_vq, phone, logits, logits.softmax(1), qp, pf, qpf, expo, pow_raw, pitch_hz, h, x,
                o["ir_amp"], o["ir_phase"], o["aperiodicity"], o["post_filter"])


def snr_db(r, e):
    r = r.astype(np.float64); e = e.astype(np.float64)
    err = np.sum((r - e) ** 2)
    return float("inf") if err == 0 else round(float(10 * np.log10(np.sum(r ** 2) / max(err, 1e-300))), 2)


def cmp(r, e):
    e = np.asarray(e, dtype=np.float32).reshape(r.shape)
    d = {"snr_db": snr_db(r, e), "max_abs_err": float(np.max(np.abs(r - e))), "ref_absmax": float(np.max(np.abs(r))),
         "finite": bool(np.isfinite(e).all())}
    return d


def convert(tr, names, shapes, outs, prec, path):
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s) for n, s in zip(names, shapes)],
                    outputs=[ct.TensorType(name=n) for n in outs], convert_to="mlprogram",
                    compute_precision=prec, minimum_deployment_target=ct.target.iOS17)
    shutil.rmtree(path, ignore_errors=True)
    ml.save(path)
    return path


def op_tests(mdir, can_run):
    """Tiny single-op models. Returns {name: {prec: result}}."""
    rng = np.random.default_rng(0)
    tests = {}

    class TopK(nn.Module):
        def forward(self, x): return x.topk(4, dim=-1)[1].float()

    class ArgMax(nn.Module):
        def forward(self, x): return x.argmax(1).float()

    class OneHotMean(nn.Module):
        def forward(self, x):
            idx = x.topk(4, dim=-1)[1]
            return F.one_hot(idx, 512).float().sum(2) * 0.25

    class CumSum(nn.Module):
        def forward(self, x): return torch.cumsum(x * x, -1)

    class Gather(nn.Module):
        def __init__(self):
            super().__init__()
            j = torch.arange(304, 560)[:, None]; m = torch.arange(304)[None, :]
            self.register_buffer("idx", (j - m).reshape(-1))
        def forward(self, x): return x[..., self.idx]

    class SoftmaxBandArgmax(nn.Module):
        def forward(self, x):
            p = x.softmax(1)
            p2 = torch.cat([torch.full_like(p[:, :1], -100.0), p[:, 1:]], 1)
            band = p2[:, :-3] + p2[:, 1:-2] + p2[:, 2:-1] + p2[:, 3:]
            return band.argmax(1).float()

    class Pow2(nn.Module):
        def forward(self, x): return 55.0 * torch.pow(2.0, x / 96.0)

    cases = {
        "topk4_idx": (TopK(), rng.standard_normal((1, 60, 512)).astype(np.float32)),
        "argmax_axis1": (ArgMax(), rng.standard_normal((1, 445, 60)).astype(np.float32)),
        "onehot_topk_mean": (OneHotMean(), rng.standard_normal((1, 60, 512)).astype(np.float32)),
        "cumsum_sq": (CumSum(), (rng.standard_normal((1, 60, 560)) * 0.1).astype(np.float32)),
        "gather_lastdim": (Gather(), rng.standard_normal((1, 60, 560)).astype(np.float32)),
        "softmax_band_argmax": (SoftmaxBandArgmax(), (rng.standard_normal((1, 448, 60)) * 4).astype(np.float32)),
        "pow2_pitch": (Pow2(), rng.integers(1, 448, (1, 60)).astype(np.float32)),
    }
    for name, (mod, x) in cases.items():
        tests[name] = {}
        xt = torch.from_numpy(x)
        with torch.no_grad():
            ref = mod(xt).numpy()
            tr = torch.jit.trace(mod.eval(), (xt,), check_trace=False)
        for prec_name, prec in [("fp32", ct.precision.FLOAT32), ("fp16", ct.precision.FLOAT16)]:
            try:
                path = convert(tr, ["x"], [x.shape], ["y"], prec, os.path.join(mdir, f"op_{name}_{prec_name}.mlpackage"))
                if not can_run:
                    tests[name][prec_name] = "converted"; continue
                y = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY).predict({"x": x})["y"]
                y = np.asarray(y, np.float32).reshape(ref.shape)
                if name == "topk4_idx":
                    eq = np.mean([set(a) == set(b) for a, b in zip(ref.reshape(-1, 4), y.reshape(-1, 4))])
                    tests[name][prec_name] = {"set_equal_pct": round(100 * float(eq), 2)}
                elif name in ("argmax_axis1", "softmax_band_argmax"):
                    tests[name][prec_name] = {"equal_pct": round(100 * float(np.mean(ref == y)), 2),
                                              "ref_head": ref.ravel()[:8].tolist(), "coreml_head": y.ravel()[:8].tolist()}
                else:
                    tests[name][prec_name] = cmp(ref, y)
            except Exception as e:  # noqa: BLE001
                tests[name][prec_name] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
    return tests


def run(pe, ps, g, se, kv, cb, out_dir, W=60, can_run=True, precs=("fp32",)):
    mdir = os.path.join(out_dir, "models_diag"); os.makedirs(mdir, exist_ok=True)
    rep = {"W": W, "stages": {}, "ops": {}, "errors": []}
    wav, sr = sf.read(os.path.join(REPO, "assets/test/common_voice_ja_38843402_16k.wav"), dtype="float32")
    n_frames = len(wav) // 160
    starts = np.linspace(0, n_frames - W, 5)[1:-1].astype(int).tolist()
    xs = [torch.from_numpy(wav[s * 160:(s + W) * 160].copy())[None, None] for s in starts]
    zero = torch.tensor([0.0])
    m = DebugNN(pe, ps, g, "safe").eval()
    with torch.no_grad():
        refs = [[t.numpy() for t in m(x, se, kv, cb, zero)] for x in xs]
        tr = torch.jit.trace(m, (xs[0], se, kv, cb, zero), check_trace=False)
    names = ["wav", "spk_embed", "kv", "codebook", "pitch_shift_bins"]
    shapes = [tuple(xs[0].shape), (1, 256), (1, 384, 128), (1, 512, 128), (1,)]
    sp = {"spk_embed": se.numpy(), "kv": kv.numpy(), "codebook": cb.numpy(), "pitch_shift_bins": np.zeros((1,), np.float32)}
    for prec_name, prec in [("fp32", ct.precision.FLOAT32), ("fp16", ct.precision.FLOAT16)]:
        if prec_name not in precs:
            continue
        try:
            t0 = time.time()
            path = convert(tr, names, shapes, STAGES, prec, os.path.join(mdir, f"debug_{prec_name}_W{W}.mlpackage"))
            print(f"diag: converted debug {prec_name} in {time.time() - t0:.1f}s"); sys.stdout.flush()
            if not can_run:
                rep["stages"][prec_name] = "converted"; continue
            ml = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
            per = {s: [] for s in STAGES}
            for x, ref in zip(xs, refs):
                y = ml.predict({"wav": x.numpy(), **sp})
                for s, r in zip(STAGES, ref):
                    if s in INT_STAGES:
                        e = np.asarray(y[s], np.float32).reshape(r.shape)
                        per[s].append({"equal_pct": round(100 * float(np.mean(e == r)), 2),
                                       "ref_head": r.ravel()[:10].tolist(), "coreml_head": e.ravel()[:10].tolist()})
                    else:
                        per[s].append(cmp(r, y[s]))
            rep["stages"][prec_name] = per
        except Exception as e:  # noqa: BLE001
            rep["errors"].append(f"debug {prec_name}: {type(e).__name__}: {str(e)[:500]}")
            traceback.print_exc()
    rep["ops"] = op_tests(mdir, can_run)
    with open(os.path.join(out_dir, "diag.json"), "w") as f:
        json.dump(rep, f, indent=1)
    md = to_md(rep)
    with open(os.path.join(out_dir, "diag.md"), "w") as f:
        f.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    print(md)
    return rep


def to_md(rep):
    L = [f"## Diagnosis (W={rep['W']}, CPU_ONLY, per stage: worst-window SNR dB / max abs err; qp: equal %)", ""]
    precs = [p for p in ("fp32", "fp16") if isinstance(rep["stages"].get(p), dict)]
    if precs:
        L += ["| stage | " + " | ".join(precs) + " |", "|---" * (len(precs) + 1) + "|"]
        for s in STAGES:
            cells = []
            for p in precs:
                v = rep["stages"][p][s]
                if s in INT_STAGES:
                    cells.append(f"{min(x['equal_pct'] for x in v)}% eq (ref {v[0]['ref_head'][:5]} cm {v[0]['coreml_head'][:5]})")
                else:
                    cells.append(f"{min(x['snr_db'] for x in v)} / {max(x['max_abs_err'] for x in v):.3g}"
                                 + ("" if all(x["finite"] for x in v) else " NaN"))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "### Single-op models (CPU_ONLY)", "", "| op | fp32 | fp16 |", "|---|---|---|"]
    for n, v in rep["ops"].items():
        def fmt(x):
            if isinstance(x, dict):
                return ", ".join(f"{k}={x[k]}" for k in x if k not in ("ref_absmax",))[:200]
            return str(x)[:200]
        L.append(f"| {n} | {fmt(v.get('fp32'))} | {fmt(v.get('fp16'))} |")
    if rep["errors"]:
        L += ["", "Errors:"] + [f"- {e}" for e in rep["errors"]]
    return "\n".join(L)
