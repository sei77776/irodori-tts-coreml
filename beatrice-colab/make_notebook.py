"""Generates BeatriceV2_AutoTrain.ipynb (run: python make_notebook.py)."""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})

def code(text, title=None):
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"} if title else {},
                  "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)})

md("""
# Beatrice V2 おまかせ学習

1. メニュー「**ランタイム → ランタイムのタイプを変更**」で **GPU**（T4 / L4 / A100）を選ぶ
2. 下の「① 設定」を選ぶ
3. メニュー「**ランタイム → すべてのセルを実行**」

あとは自動で、学習ツールの準備 → 音声の取り込み → 学習 → 結果の保存まで進みます。Google ドライブに保存する設定（既定）では、途中で接続が切れても、もう一度「すべてのセルを実行」すれば続きから再開します。

- 学習ツール: [fierce-cats/beatrice-trainer](https://huggingface.co/fierce-cats/beatrice-trainer)（MIT License, Project Beatrice）
- 「つくよみちゃんコーパス」を選んだ場合: 音声は[公式配布元](https://tyc.rei-yumesaki.net/material/corpus/)から直接取得します（再配布はしません）。変換音声を公開するときは「フリー素材キャラクター「つくよみちゃん」が無料公開している音声データを使用しています」と表記してください。
- 自分の音声を使う場合は、本人の許可がある声だけを使ってください。
""")

code(r'''
#@title ① 設定
VOICE_SOURCE = "つくよみちゃんコーパス（自動ダウンロード）" #@param ["つくよみちゃんコーパス（自動ダウンロード）", "自分の音声 zip をアップロード"]
SPEAKER_NAME = "tsukuyomi" #@param {type:"string"}
TRAINING_STEPS = 10000 #@param {type:"integer"}
SAVE_TO_GOOGLE_DRIVE = True #@param {type:"boolean"}
''', title=True)

code(r'''
#@title ② 準備（GPU の確認・保存先・学習ツールの取得）
import os, sys, re, glob, json, shutil, zipfile, subprocess, urllib.request
import torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU が使えません。「ランタイム → ランタイムのタイプを変更」で GPU を選び、もう一度「すべてのセルを実行」してください。")
print("GPU:", torch.cuda.get_device_name(0))

SPEAKER_NAME = re.sub(r"[^A-Za-z0-9_-]", "_", SPEAKER_NAME.strip()) or "speaker"
WORK = "/content/beatrice-trainer"
DATA = "/content/beatrice-data"
if SAVE_TO_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = f"/content/drive/MyDrive/BeatriceTraining/{SPEAKER_NAME}"
else:
    BASE = f"/content/BeatriceTraining/{SPEAKER_NAME}"
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
print("保存先:", BASE)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile", "pyworld", "huggingface_hub"], check=True)
from huggingface_hub import snapshot_download
snapshot_download("fierce-cats/beatrice-trainer", local_dir=WORK)
os.makedirs(os.path.join(WORK, ".git"), exist_ok=True)  # 学習コードは .git の有無でリポジトリの場所を判定する

# Colab の torchaudio (>=2.9) は音声 I/O を削除しているため、読み込みを soundfile に置き換える（学習内容は不変）
main_py = os.path.join(WORK, "beatrice_trainer", "__main__.py")
src = open(main_py, encoding="utf-8").read()
old = 'assert "soundfile" in torchaudio.list_audio_backends()\n'
new = (
    "import soundfile as _sf\n"
    "def _load_with_soundfile(path, *args, **kwargs):\n"
    "    data, sr = _sf.read(str(path), dtype='float32', always_2d=True)\n"
    "    return torch.from_numpy(data.T.copy()), sr\n"
    "torchaudio.load = _load_with_soundfile\n"
)
if old in src:
    open(main_py, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print("学習コードを Colab 用に調整しました")
elif "_load_with_soundfile" in src:
    print("学習コードは調整済みです")
else:
    print("注意: 想定した行が見つかりません（学習コードが更新された可能性があります）")
''', title=True)

code(r'''
#@title ③ 音声の取り込み
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
spk_dir = os.path.join(DATA, SPEAKER_NAME)
shutil.rmtree(DATA, ignore_errors=True)
os.makedirs(spk_dir)

def take_audio(zf, decode_name=lambda n: n, accept=lambda n: True):
    count = 0
    for info in zf.infolist():
        name = decode_name(info.filename)
        base = os.path.basename(name)
        if info.is_dir() or "__MACOSX" in name or base.startswith(".") or not base.lower().endswith(AUDIO_EXT):
            continue
        if not accept(name):
            continue
        count += 1
        dest = os.path.join(spk_dir, f"{count:05d}_{base}")
        with zf.open(info) as s, open(dest, "wb") as d:
            shutil.copyfileobj(s, d)
    return count

if VOICE_SOURCE.startswith("つくよみ"):
    zpath = "/content/tyc-corpus1.zip"
    if not os.path.exists(zpath):
        print("つくよみちゃんコーパスを公式配布元からダウンロードしています…")
        urllib.request.urlretrieve("https://tyc.rei-yumesaki.net/files/voice/tyc-corpus1.zip", zpath)
    def sjis(n):
        try:
            return n.encode("cp437").decode("cp932")
        except UnicodeError:
            return n
    with zipfile.ZipFile(zpath) as zf:
        n = take_audio(zf, sjis, lambda name: "02 WAV" in name)  # 「+12dB 増幅」版の 100 文
else:
    from google.colab import files
    print("学習に使う音声の zip を選んでください（中のフォルダ構成は自由。音声ファイルだけを使います）")
    uploaded = files.upload()
    n = 0
    for fname in uploaded:
        with zipfile.ZipFile(fname) as zf:
            n += take_audio(zf)
if n == 0:
    raise RuntimeError("音声ファイルが見つかりませんでした。")
print(f"{n} 個の音声ファイルを取り込みました → {spk_dir}")
''', title=True)

code(r'''
#@title ④ 学習（自動で再開にも対応）
cfg = json.load(open(os.path.join(WORK, "assets", "default_config.json"), encoding="utf-8"))
cfg["n_steps"] = int(TRAINING_STEPS)
cfg_path = os.path.join(BASE, "config_run.json")
json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2)
resume = os.path.exists(os.path.join(OUT, "checkpoint_latest.pt.gz"))
print("前回の続きから再開します" if resume else "最初から学習します", f"（{cfg['n_steps']} ステップ）")
cmd = [sys.executable, "beatrice_trainer", "-d", DATA, "-o", OUT, "-c", cfg_path] + (["-r"] if resume else [])
proc = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    raise RuntimeError("学習が途中で止まりました。上のエラーを確認してください（もう一度「すべてのセルを実行」すると続きから再開します）。")
''', title=True)

code(r'''
#@title ⑤ 結果のまとめ（Windows 用モデル と Claude に渡すチェックポイント）
def step_of(p):
    m = re.search(r"_(\d+)$", p.rstrip("/"))
    return int(m.group(1)) if m else -1
paras = sorted((p for p in glob.glob(os.path.join(OUT, "paraphernalia_*")) if os.path.isdir(p)), key=step_of)
ckpt = os.path.join(OUT, "checkpoint_latest.pt.gz")
if not paras or not os.path.exists(ckpt):
    raise RuntimeError("学習結果が見つかりません。④がエラーなく終わったか確認してください。")
latest = paras[-1]
step = step_of(latest)
para_zip = os.path.join(BASE, f"{SPEAKER_NAME}_paraphernalia_{step}.zip")
with zipfile.ZipFile(para_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, fs in os.walk(latest):
        for f in fs:
            full = os.path.join(root, f)
            zf.write(full, os.path.relpath(full, os.path.dirname(latest)))
ckpt_copy = os.path.join(BASE, f"{SPEAKER_NAME}_checkpoint_{step}.pt.gz")
shutil.copy(ckpt, ckpt_copy)
print("できました:")
print(" Windows（beatrice-client）用:", para_zip)
print(" Claude に渡すチェックポイント:", ckpt_copy)
if SAVE_TO_GOOGLE_DRIVE:
    print("Google ドライブの「BeatriceTraining/%s」フォルダに保存しました。" % SPEAKER_NAME)
else:
    from google.colab import files
    files.download(para_zip)
    files.download(ckpt_copy)
''', title=True)

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open("BeatriceV2_AutoTrain.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote BeatriceV2_AutoTrain.ipynb", len(cells), "cells")
