"""Beatrice v2 Core ML: conversion + numeric agreement + per-call latency.

Self-contained: needs only the beatrice-trainer clone under $BEATRICE_WORK (see run.sh).

  python bench.py --out results            # convert, compare and benchmark (macOS)
  python bench.py --out results --convert-only   # Linux: conversion check only

Models (one pseudo-streaming call over a fixed window of W frames = 10 ms each):
  split_W{W}           : pitch features FP32 (pitchfeat) + all networks FP16 (main), pitch_hz via pow
  v2_*_W{W}            : same split, main = StreamNNv2 (pitch_hz via lookup table, extra qp output)
                         v2_fp32 / v2_fp16 / v2_isl (VQ, sample_pitch, pitch_hz in FP32)
                         v2_isl_pnet (+ pitch net FP32) / v2_isl_phone_pnet (+ phone extractor FP32)
  (BEATRICE_MODELS=split,v2_isl ... restricts the set; BEATRICE_DIAG=0 skips diag.py)
Reference: the trainer's own modules in PyTorch FP32 (StreamNN variant "orig").
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_infer import REPO, load_models  # noqa: E402
from coreml_wrap import PitchFeaturesSafe, StreamNN, StreamNNv2, make_fp32_selector, speaker_inputs  # noqa: E402

import coremltools as ct  # noqa: E402

warnings.filterwarnings("ignore")
OUT_NAMES = ["ir_amp", "ir_phase", "aperiodicity", "post_filter", "pitch_hz"]
TEST_WAV = "assets/test/common_voice_ja_38843402_16k.wav"
COMPUTE_UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]


def host_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "coremltools": ct.__version__, "numpy": np.__version__}
    for key, cmd in [("cpu_brand", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                     ("hw_model", ["sysctl", "-n", "hw.model"]),
                     ("ncpu", ["sysctl", "-n", "hw.ncpu"]),
                     ("memsize", ["sysctl", "-n", "hw.memsize"]),
                     ("vmm_present", ["sysctl", "-n", "kern.hv_vmm_present"])]:
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # noqa: BLE001
            info[key] = f"n/a ({type(e).__name__})"
    return info


class Feats(nn.Module):
    def __init__(self):
        super().__init__()
        self.pf = PitchFeaturesSafe()

    def forward(self, wav):
        return self.pf(wav.squeeze(1))


class Main(nn.Module):
    """StreamNN('safe') with the pitch features given as inputs."""

    def __init__(self, b):
        super().__init__()
        self.b = b

    def forward(self, wav, inst, corr, energy, spk_embed, kv, codebook, pitch_shift_bins):
        b = self.b
        p = b.ps
        b.pitch = lambda _x: (p.head(p.backbone(F.gelu(
            p.instfreq_embed_1(F.gelu(p.instfreq_embed_0(inst), approximate="tanh"))
            + p.corr_embed_1(F.gelu(p.corr_embed_0(corr), approximate="tanh")), approximate="tanh"))), energy)
        return b.forward(wav, spk_embed, kv, codebook, pitch_shift_bins)


class MainV2(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, wav, inst, corr, energy, spk_embed, kv, codebook, pitch_shift_bins):
        return self.v.forward_pf(wav, inst, corr, energy, spk_embed, kv, codebook, pitch_shift_bins)


def pkg_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def convert(trace, names, shapes, out_names, precision, path):
    t0 = time.time()
    ml = ct.convert(trace, inputs=[ct.TensorType(name=n, shape=s) for n, s in zip(names, shapes)],
                    outputs=[ct.TensorType(name=n) for n in out_names], convert_to="mlprogram",
                    compute_precision=precision, minimum_deployment_target=ct.target.iOS17)
    shutil.rmtree(path, ignore_errors=True)
    ml.save(path)
    n_ops = sum(len(b.operations) for f in ml.get_spec().mlProgram.functions.values()
                for b in f.block_specializations.values())
    return {"path": path, "size_mb": round(pkg_size(path) / 1e6, 2), "convert_s": round(time.time() - t0, 1),
            "n_ops": n_ops}


def snr_db(ref, est):
    ref = ref.astype(np.float64); est = est.astype(np.float64)
    err = np.sum((ref - est) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(ref ** 2) / err))


def pitch_bins(p):
    return np.round(np.log2(np.maximum(p, 1e-3) / 55.0) * 96.0)


def compare(ref, out):
    res = {}
    for n in OUT_NAMES:
        r = ref[n]; o = np.asarray(out[n], dtype=np.float32).reshape(r.shape)
        res[n] = {"snr_db": round(snr_db(r, o), 2), "max_abs_err": float(np.max(np.abs(r - o))),
                  "finite": bool(np.isfinite(o).all())}
    qr, qo = pitch_bins(ref["pitch_hz"]), pitch_bins(np.asarray(out["pitch_hz"]).reshape(ref["pitch_hz"].shape))
    d = np.abs(qr - qo)
    res["pitch_bin_equal_pct"] = round(100 * float(np.mean(d == 0)), 2)
    res["pitch_frames_gt_1semitone"] = int(np.sum(d > 8))
    res["pitch_frames"] = int(d.size)
    return res


def merge_cmp(lst):
    """Aggregate per-window comparisons: worst SNR, max error, mean bin equality."""
    agg = {}
    for n in OUT_NAMES:
        agg[n] = {"snr_db_min": min(c[n]["snr_db"] for c in lst),
                  "max_abs_err": max(c[n]["max_abs_err"] for c in lst),
                  "finite": all(c[n]["finite"] for c in lst)}
    agg["pitch_bin_equal_pct"] = round(float(np.mean([c["pitch_bin_equal_pct"] for c in lst])), 2)
    agg["pitch_frames_gt_1semitone"] = sum(c["pitch_frames_gt_1semitone"] for c in lst)
    agg["pitch_frames"] = sum(c["pitch_frames"] for c in lst)
    return agg


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"median_ms": round(float(np.median(ts)), 2), "p90_ms": round(float(np.percentile(ts, 90)), 2),
            "min_ms": round(float(ts.min()), 2), "iters": iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--windows", default="60,112")
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--convert-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mdir = os.path.join(a.out, "models"); os.makedirs(mdir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    report = {"host": host_info(), "note": "", "models": {}, "accuracy": {}, "latency": {}, "errors": []}
    if report["host"].get("vmm_present") == "1":
        report["note"] = ("Runner is a virtual machine (kern.hv_vmm_present=1): the Neural Engine is normally not "
                          "exposed to macOS VMs and GPU may be paravirtualized, so CPU_AND_NE / ALL / CPU_AND_GPU "
                          "may silently fall back to CPU or be slower than on a real device.")
    print(json.dumps(report["host"], indent=1))

    pe, ps, g = load_models()
    se, kv, cb = speaker_inputs(g, 0)
    if os.environ.get("BEATRICE_DIAG", "1") != "0":
        import diag
        try:
            diag.run(pe, ps, g, se, kv, cb, a.out, W=60, can_run=not a.convert_only, precs=("fp32",))
        except Exception as e:  # noqa: BLE001
            report["errors"].append({"stage": "diag", "error": f"{type(e).__name__}: {e}"[:2000]})
            traceback.print_exc()
    ref_model = StreamNN(pe, ps, g, "orig").eval()
    safe = StreamNN(pe, ps, g, "safe").eval()
    wav, sr = sf.read(os.path.join(REPO, TEST_WAV), dtype="float32")
    assert sr == 16000
    n_frames = len(wav) // 160
    zero = torch.tensor([0.0])

    for W in [int(w) for w in a.windows.split(",")]:
        starts = np.linspace(0, n_frames - W, a.n_windows + 2)[1:-1].astype(int).tolist()
        xs = [torch.from_numpy(wav[s * 160:(s + W) * 160].copy())[None, None] for s in starts]
        # ---- PyTorch reference (FP32, trainer modules) + PyTorch eager timing
        refs = []
        with torch.inference_mode():
            for x in xs:
                refs.append({n: t.numpy() for n, t in zip(OUT_NAMES, ref_model(x, se, kv, cb, zero))})
            report["latency"][f"pytorch_eager_fp32_W{W}"] = {
                "CPU": timeit(lambda: safe(xs[0], se, kv, cb, zero), a.warmup, min(a.iters, 20))}
        # ---- convert
        fm, mm = Feats().eval(), Main(StreamNN(pe, ps, g, "safe").eval()).eval()
        x0 = xs[0]
        with torch.no_grad():
            inst, corr, en = fm(x0)
            tr_single = torch.jit.trace(safe, (x0, se, kv, cb, zero), check_trace=False)
            tr_feat = torch.jit.trace(fm, (x0,), check_trace=False)
            tr_main = torch.jit.trace(mm, (x0, inst, corr, en, se, kv, cb, zero), check_trace=False)
            tr_v2 = torch.jit.trace(MainV2(StreamNNv2(pe, ps, g).eval()).eval(), (x0, inst, corr, en, se, kv, cb, zero),
                                    check_trace=False)
        single_in = ["wav", "spk_embed", "kv", "codebook", "pitch_shift_bins"]
        single_sh = [tuple(x0.shape), (1, 256), (1, 384, 128), (1, 512, 128), (1,)]
        main_in = ["wav", "inst", "corr", "energy"] + single_in[1:]
        main_sh = [tuple(x0.shape), tuple(inst.shape), tuple(corr.shape), tuple(en.shape)] + single_sh[1:]
        feat = (tr_feat, ["wav"], [tuple(x0.shape)], ["inst", "corr", "energy"], ct.precision.FLOAT32)
        v2_out = OUT_NAMES + ["qp"]

        def mixed(regions):
            return ct.transform.FP16ComputePrecision(op_selector=make_fp32_selector(regions))
        # v2 main: pitch_hz via lookup table (+ qp output); FP32 islands selected by region
        specs = {
            f"split_W{W}": [feat, (tr_main, main_in, main_sh, OUT_NAMES, ct.precision.FLOAT16)],
            f"v2_fp32_W{W}": [feat, (tr_v2, main_in, main_sh, v2_out, ct.precision.FLOAT32)],
            f"v2_fp16_W{W}": [feat, (tr_v2, main_in, main_sh, v2_out, ct.precision.FLOAT16)],
            f"v2_isl_W{W}": [feat, (tr_v2, main_in, main_sh, v2_out, mixed(("vq", "sp", "hz")))],
            f"v2_isl_pnet_W{W}": [feat, (tr_v2, main_in, main_sh, v2_out, mixed(("vq", "sp", "hz", "pnet")))],
            f"v2_isl_phone_pnet_W{W}": [feat, (tr_v2, main_in, main_sh, v2_out,
                                              mixed(("phone", "vq", "sp", "hz", "pnet")))],
        }
        only = os.environ.get("BEATRICE_MODELS")
        if only:
            specs = {k: v for k, v in specs.items() if any(k.startswith(o + "_W") for o in only.split(","))}
        for name, parts in specs.items():
            try:
                infos = []
                for i, (tr, ins, shs, outs, prec) in enumerate(parts):
                    sub = ["", "pitchfeat_fp32", "main"][0 if len(parts) == 1 else i + 1]
                    path = os.path.join(mdir, f"beatrice_{name}{'_' + sub if sub else ''}.mlpackage")
                    infos.append(convert(tr, ins, shs, outs, prec, path))
                report["models"][name] = infos
                print("converted", name, [(i["size_mb"], i["n_ops"]) for i in infos]); sys.stdout.flush()
            except Exception as e:  # noqa: BLE001
                report["errors"].append({"stage": f"convert {name}", "error": f"{type(e).__name__}: {e}"[:2000]})
                traceback.print_exc()
        if a.convert_only:
            continue
        # ---- Core ML: accuracy + latency per compute unit
        sp = {"spk_embed": se.numpy(), "kv": kv.numpy(), "codebook": cb.numpy(),
              "pitch_shift_bins": np.zeros((1,), np.float32)}
        for name, infos in list(report["models"].items()):
            if not name.endswith(f"_W{W}"):
                continue
            for cu in COMPUTE_UNITS:
                key = f"{name}/{cu}"
                try:
                    t0 = time.perf_counter()
                    mls = [ct.models.MLModel(i["path"], compute_units=getattr(ct.ComputeUnit, cu)) for i in infos]
                    load_ms = (time.perf_counter() - t0) * 1000
                    if len(mls) == 1:
                        def run(x, mls=mls):
                            return mls[0].predict({"wav": x.numpy(), **sp})
                    else:
                        def run(x, mls=mls):
                            f = mls[0].predict({"wav": x.numpy()})
                            return mls[1].predict({"wav": x.numpy(), "inst": f["inst"], "corr": f["corr"],
                                                   "energy": f["energy"], **sp})
                    cmps = [compare(r, run(x)) for x, r in zip(xs, refs)]
                    report["accuracy"][key] = merge_cmp(cmps)
                    lat = {"load_ms": round(load_ms, 1), "total": timeit(lambda: run(x0), a.warmup, a.iters)}
                    if len(mls) == 2:
                        f0 = mls[0].predict({"wav": x0.numpy()})
                        lat["pitchfeat_only"] = timeit(lambda: mls[0].predict({"wav": x0.numpy()}), a.warmup, a.iters)
                        main_in_d = {"wav": x0.numpy(), "inst": f0["inst"], "corr": f0["corr"], "energy": f0["energy"], **sp}
                        lat["main_only"] = timeit(lambda: mls[1].predict(main_in_d), a.warmup, a.iters)
                    lat["chunk_budget_note"] = "real time needs total < chunk length (e.g. 50 or 100 ms per call)"
                    report["latency"][key] = lat
                    print(key, report["accuracy"][key]["pitch_bin_equal_pct"], lat["total"]); sys.stdout.flush()
                except Exception as e:  # noqa: BLE001
                    report["errors"].append({"stage": key, "error": f"{type(e).__name__}: {e}"[:2000]})
                    traceback.print_exc()

    with open(os.path.join(a.out, "bench.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    md = to_markdown(report)
    with open(os.path.join(a.out, "bench.md"), "w") as f:
        f.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    print(md)


def to_markdown(r):
    h = r["host"]
    L = ["## Beatrice Core ML bench", "",
         f"- host: `{h.get('hw_model')}` / `{h.get('cpu_brand')}` / ncpu {h.get('ncpu')} / VM={h.get('vmm_present')} / {h.get('platform')}",
         f"- torch {h['torch']}, coremltools {h['coremltools']}, numpy {h['numpy']}"]
    if r["note"]:
        L.append(f"- **note**: {r['note']}")
    L += ["", "### Models", "", "| model | part | size MB | ops | convert s |", "|---|---|---|---|---|"]
    for n, infos in r["models"].items():
        for i in infos:
            L.append(f"| {n} | {os.path.basename(i['path'])} | {i['size_mb']} | {i['n_ops']} | {i['convert_s']} |")
    if r["accuracy"]:
        L += ["", "### Accuracy vs PyTorch FP32 (worst window SNR dB / max abs err)", "",
              "| model / compute unit | " + " | ".join(OUT_NAMES) + " | pitch bin equal % | frames >1 semitone |",
              "|---" * (len(OUT_NAMES) + 3) + "|"]
        for k, v in r["accuracy"].items():
            cells = [f"{v[n]['snr_db_min']:.1f} / {v[n]['max_abs_err']:.2g}" + ("" if v[n]["finite"] else " NaN!")
                     for n in OUT_NAMES]
            L.append(f"| {k} | " + " | ".join(cells) +
                     f" | {v['pitch_bin_equal_pct']} | {v['pitch_frames_gt_1semitone']}/{v['pitch_frames']} |")
    if r["latency"]:
        L += ["", "### Latency per call (ms, median / p90)", "",
              "| model / compute unit | load ms | total | pitchfeat only | main only |", "|---|---|---|---|---|"]
        for k, v in r["latency"].items():
            def f(x):
                return f"{x['median_ms']} / {x['p90_ms']}" if isinstance(x, dict) else "-"
            if "total" in v:
                L.append(f"| {k} | {v['load_ms']} | {f(v['total'])} | {f(v.get('pitchfeat_only'))} | {f(v.get('main_only'))} |")
            else:
                L.append(f"| {k} | - | {f(v['CPU'])} | - | - |")
    if r["errors"]:
        L += ["", "### Errors", ""] + [f"- `{e['stage']}`: {e['error'][:300]}" for e in r["errors"]]
    return "\n".join(L)


if __name__ == "__main__":
    main()
