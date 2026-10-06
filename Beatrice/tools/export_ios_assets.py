"""Export everything the iOS/macOS real-time voice-conversion prototype needs.

  BEATRICE_WORK=<dir with beatrice-trainer clone> python -I export_ios_assets.py \
      --assets ../../Examples/BeatriceAssets --golden ../../Tests/BeatriceVCTests/Golden

Outputs
  <assets>/BeatriceFP32.mlpackage   multifunction (w64 / w116 / w216), all FP32
  <assets>/BeatriceMixed.mlpackage  multifunction, FP16 with FP32 islands (pitch features, VQ,
                                    sample_pitch, pitch_hz, pitch net)
  <assets>/voices.json, voice_XXX.bin, common.bin
  <golden>/...                      reference data for the Swift tests (DSP, model, streaming)

Window W = left context + chunk + lookahead (4 frames = 40 ms), W % 4 == 0 (phone-extractor
attention). The app picks a function by context length and derives the left context from the chunk:
  w64  -> ~0.5 s, w116 -> ~1 s, w216 -> ~2 s.

To use a speaker trained on Colab, point BEATRICE_NET_G at the fine-tuned checkpoint
(checkpoint_latest.pt.gz) and pass --speakers all: the vocoder weights change with fine-tuning,
so the models must be re-exported together with the voices.

Voice packs: the app loads every directory under Examples/BeatriceAssets that has a voices.json
(the root = the pretrained pack). Export a fine-tuned speaker into its own subdirectory and describe
it with --pack-json (merged into voices.json: "pack" {id, name, order, voice_credit, terms} and
"credits"). --models limits the exported precisions; voices.json lists only the models present.
The Tsukuyomi-chan pack was made with:

  BEATRICE_NET_G=tsukuyomi_checkpoint_10000.pt.gz python -I export_ios_assets.py \
      --assets ../../Examples/BeatriceAssets/tsukuyomi --golden ../../Tests/BeatriceVCTests/GoldenTsukuyomi \
      --speakers all --label つくよみちゃん --models BeatriceFP32 \
      --pack-json packs/tsukuyomi.json
"""
import argparse
import json
import os
import shutil
import sys
import warnings

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_infer import REPO, dsp_part, load_models  # noqa: E402
from coreml_wrap import StreamNNv3, make_fp32_selector, speaker_inputs  # noqa: E402

import coremltools as ct  # noqa: E402

warnings.filterwarnings("ignore")
FUNCTIONS = {"w64": 64, "w116": 116, "w216": 216}
LOOKAHEAD = 4
CROSSFADE = 2
HOP_IN, HOP_OUT = 160, 240
IN_NAMES = ["wav", "spk_embed", "kv", "codebook", "pitch_shift_bins"]
OUT_NAMES = ["ir_amp", "ir_phase", "aperiodicity", "post_filter", "pitch_hz", "qp"]
TEST_WAV = "assets/test/common_voice_ja_38843402_16k.wav"


# ---------------------------------------------------------------- noise (shared with Swift)
def noise_at(idx):
    """Uniform noise in [-0.5, 0.5) as a pure function of the absolute 24 kHz sample index
    (SplitMix64, top 24 bits). Swift: BeatriceNoise.value(at:)."""
    with np.errstate(over="ignore"):
        z = np.asarray(idx, dtype=np.int64).astype(np.uint64)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(40)).astype(np.float64) / float(1 << 24) - 0.5).astype(np.float32)


# ---------------------------------------------------------------- reference streaming
def run_nn(m, wav_window, se, kv, cb, psb):
    with torch.no_grad():
        o = m(torch.from_numpy(wav_window)[None, None], se, kv, cb, torch.tensor([float(psb)]))
    return {n: t for n, t in zip(OUT_NAMES, o)}


def dsp_window(g, o, init_phase, exc):
    p = {"ir_amp": o["ir_amp"], "ir_phase": o["ir_phase"], "aperiodicity": o["aperiodicity"],
         "post_filter": o["post_filter"], "pitch": o["pitch_hz"]}
    y, _, _, cum = dsp_part(g.vocoder, p, init_phase, torch.from_numpy(exc)[None])
    return y[0].numpy(), cum[0].numpy()


def stream_reference(m, g, wav, W, c, se, kv, cb, psb=0.0, n_chunks=None):
    """Canonical streaming algorithm (BeatriceStreamer.swift implements the same):
    history starts as zeros, each chunk shifts it; output = window frames [C, C+c) with
    pulse-phase carry and a CROSSFADE-frame linear crossfade into the previous look-ahead tail."""
    C = W - c - LOOKAHEAD
    a, b = C * HOP_OUT, (C + c) * HOP_OUT
    hist = np.zeros(W * HOP_IN, np.float32)
    phi, prev, out = 0.0, None, []
    fade = np.linspace(0.0, 1.0, CROSSFADE * HOP_OUT, dtype=np.float32)
    total = len(wav) // (c * HOP_IN)
    n_chunks = total if n_chunks is None else min(n_chunks, total)
    for k in range(n_chunks):
        hist = np.concatenate([hist[c * HOP_IN:], wav[k * c * HOP_IN:(k + 1) * c * HOP_IN]])
        ws = (k + 1) * c - W  # global frame index of window frame 0
        o = run_nn(m, hist, se, kv, cb, psb)
        exc = noise_at(np.arange(ws * HOP_OUT, (ws + W + 1) * HOP_OUT))
        nf = np.repeat(o["pitch_hz"][0].numpy().astype(np.float32), HOP_OUT) / np.float32(24000.0)
        s_before = float(np.sum(nf[1:a].astype(np.float64)))
        init = (phi - s_before) % 1.0
        y, cum = dsp_window(g, o, init, exc)
        phi = float(cum[b - 1]) % 1.0
        seg = y[a:b].copy()
        if prev is not None:
            seg[:CROSSFADE * HOP_OUT] = prev * (1 - fade) + seg[:CROSSFADE * HOP_OUT] * fade
        prev = y[b:b + CROSSFADE * HOP_OUT].copy()
        out.append(seg)
    return np.concatenate(out)


# ---------------------------------------------------------------- export
def pick_speakers(g, n):
    e = torch.nn.functional.normalize(g.embed_speaker.weight.detach(), dim=1)
    chosen = [0]  # deterministic start, then farthest-point sampling by cosine similarity
    while len(chosen) < n:
        sim = (e @ e[chosen].T).max(1).values
        sim[chosen] = 9.0
        chosen.append(int(torch.argmin(sim)))
    return chosen


def convert_function(tr, W, precision, path):
    shapes = [(1, 1, W * HOP_IN), (1, 256), (1, 384, 128), (1, 512, 128), (1,)]
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in zip(IN_NAMES, shapes)],
                    outputs=[ct.TensorType(name=n, dtype=np.float32) for n in OUT_NAMES],
                    convert_to="mlprogram", compute_precision=precision,
                    minimum_deployment_target=ct.target.iOS18)
    shutil.rmtree(path, ignore_errors=True)
    ml.save(path)


def pkg_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def export_models(pe, ps, g, se, kv, cb, assets, tmp, which):
    m = StreamNNv3(pe, ps, g).eval()
    def precision(name):
        if name == "BeatriceFP32":
            return ct.precision.FLOAT32
        return ct.transform.FP16ComputePrecision(op_selector=make_fp32_selector(("pf", "vq", "sp", "hz", "pnet")))
    info = {}
    for name in ("BeatriceFP32", "BeatriceMixed"):
        if which and name not in which:
            continue
        desc = ct.utils.MultiFunctionDescriptor()
        for fn, W in FUNCTIONS.items():
            x = torch.zeros(1, 1, W * HOP_IN)
            with torch.no_grad():
                tr = torch.jit.trace(m, (x, se, kv, cb, torch.tensor([0.0])), check_trace=False)
            p = os.path.join(tmp, f"{name}_{fn}.mlpackage")
            convert_function(tr, W, precision(name), p)
            desc.add_function(p, src_function_name="main", target_function_name=fn)
            print(f"converted {name}/{fn} ({pkg_size(p) / 1e6:.1f} MB)", flush=True)
        desc.default_function_name = "w116"
        out = os.path.join(assets, f"{name}.mlpackage")
        shutil.rmtree(out, ignore_errors=True)
        ct.utils.save_multifunction(desc, out)
        info[name] = round(pkg_size(out) / 1e6, 2)
        print(f"saved {out}: {info[name]} MB", flush=True)
    return info


def export_voices(g, ids, assets, label, extra=None, models=("BeatriceFP32", "BeatriceMixed")):
    os.makedirs(assets, exist_ok=True)
    voices = []
    for i in ids:
        se = g.embed_speaker.weight[i].detach().float().numpy()
        kv = g.key_value_speaker_embedding.weight[i].detach().float().numpy()
        cb = g.vq.codebooks[i].detach().float().numpy().reshape(-1)
        fn = f"voice_{i:03d}.bin"
        np.concatenate([se, kv, cb]).astype("<f4").tofile(os.path.join(assets, fn))
        name = label if len(ids) == 1 else f"{label} #{i}"
        voices.append({"id": i, "name": name, "file": fn})
    common = np.concatenate([g.vocoder.ir_window.detach().float().numpy(),
                             g.embed_formant_shift.weight.detach().float().numpy().reshape(-1)])
    common.astype("<f4").tofile(os.path.join(assets, "common.bin"))
    meta = {
        "format": 1,
        "voices": voices,
        "common": {"file": "common.bin", "ir_window": 512, "formant_shift": [9, 256]},
        "voice_layout": {"spk_embed": 256, "kv": [384, 128], "codebook": [512, 128]},
        "functions": FUNCTIONS,
        "lookahead_frames": LOOKAHEAD,
        "crossfade_frames": CROSSFADE,
        "models": {k: f"{n}.mlpackage" for k, n in (("fp32", "BeatriceFP32"), ("mixed", "BeatriceMixed"))
                   if n in models},
        "credits": [
            "Voice conversion model: Beatrice v2 (beatrice-trainer 2.0.0-rc.0, fierce-cats, MIT License)",
            "Pretrained speakers: trained on LibriTTS-R (CC BY 4.0) and other corpora listed in beatrice-trainer assets/README.md",
        ],
    }
    meta.update(extra or {})
    with open(os.path.join(assets, "voices.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def save_f32(d, name, arr):
    np.asarray(arr, dtype="<f4").reshape(-1).tofile(os.path.join(d, name + ".f32"))


def export_golden(pe, ps, g, spk, golden):
    os.makedirs(golden, exist_ok=True)
    m = StreamNNv3(pe, ps, g).eval()
    se, kv, cb = speaker_inputs(g, spk)
    wav, sr = sf.read(os.path.join(REPO, TEST_WAV), dtype="float32")
    assert sr == 16000
    W = 64
    # 1) one window: model reference + DSP reference
    start = 150
    xw = wav[start * HOP_IN:(start + W) * HOP_IN].copy()
    o = run_nn(m, xw, se, kv, cb, 0.0)
    ws = 1234
    exc = noise_at(np.arange(ws * HOP_OUT, (ws + W + 1) * HOP_OUT))
    init = 0.3
    y, cum = dsp_window(g, o, init, exc)
    save_f32(golden, "nn_wav", xw)
    for n in OUT_NAMES:
        save_f32(golden, "nn_" + n, o[n][0].numpy())
    save_f32(golden, "dsp_excitation", exc)
    save_f32(golden, "dsp_out", y)
    # 2) streaming: 2.4 s, 100 ms chunks, W = 64
    c = 10
    n_chunks = 24
    sin = wav[:n_chunks * c * HOP_IN].copy()
    sout = stream_reference(m, g, sin, W, c, se, kv, cb, 0.0, n_chunks)
    save_f32(golden, "stream_in", sin)
    save_f32(golden, "stream_out", sout)
    meta = {"W": W, "speaker": spk, "nn_window_start_frame": start, "dsp_init_phase": init,
            "dsp_window_start_frame": ws, "noise_head": noise_at(np.arange(5)).tolist(),
            "stream": {"chunk": c, "lookahead": LOOKAHEAD, "crossfade": CROSSFADE, "chunks": n_chunks,
                       "pitch_shift_bins": 0.0, "formant_index": 4}}
    with open(os.path.join(golden, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("golden written:", golden, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--tmp", default="/tmp/beatrice_export")
    ap.add_argument("--speakers", default="auto5", help="autoN | all | comma-separated ids")
    ap.add_argument("--label", default="話者")
    ap.add_argument("--pack-json", help="JSON merged into voices.json (pack metadata, credits)")
    ap.add_argument("--models", default="", help="comma list of BeatriceFP32,BeatriceMixed (default both)")
    ap.add_argument("--skip-models", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    pe, ps, g = load_models()
    n_spk = g.embed_speaker.weight.size(0)
    if a.speakers.startswith("auto"):
        ids = pick_speakers(g, int(a.speakers[4:] or 5))
    elif a.speakers == "all":
        ids = list(range(n_spk))
    else:
        ids = [int(s) for s in a.speakers.split(",")]
    print("speakers:", ids, flush=True)
    which = set(filter(None, a.models.split(",")))
    extra = None
    if a.pack_json:
        with open(a.pack_json, encoding="utf-8") as f:
            extra = json.load(f)
    present = [n for n in ("BeatriceFP32", "BeatriceMixed")
               if (not which or n in which) or os.path.isdir(os.path.join(a.assets, n + ".mlpackage"))]
    export_voices(g, ids, a.assets, a.label, extra, present)
    se, kv, cb = speaker_inputs(g, ids[0])
    if not a.skip_models:
        info = export_models(pe, ps, g, se, kv, cb, a.assets, a.tmp, which)
        print("model sizes MB:", info)
    if a.golden:
        export_golden(pe, ps, g, ids[0], a.golden)


if __name__ == "__main__":
    main()
