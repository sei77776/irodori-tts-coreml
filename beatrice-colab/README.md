# Beatrice V2 おまかせ学習（Colab）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sei77776/irodori-tts-coreml/blob/claude/beatrice-bench/beatrice-colab/BeatriceV2_AutoTrain.ipynb)

GPU を選んで「すべてのセルを実行」するだけで、学習ツールの準備 → 音声の取り込み（つくよみちゃんコーパス or 自分の zip）→ 学習 → 結果の保存まで進む。Google ドライブ保存時は中断後に再実行で続きから再開。

- 生成: `python make_notebook.py`（ノートブックはこのスクリプトから作る。直接編集しない）
- Colab 向けの調整: torchaudio>=2.9 で消えた音声 I/O を soundfile に置換、`snapshot_download` 後に `.git` を作成（学習コードがリポジトリ位置を `.git` で判定するため）
- 検証（2026-10-06）: Linux CPU で n_steps=2 → 3 の通し実行と自動再開を確認。音質評価モデル（torch.hub の SpeechMOS）だけは検証環境から GitHub API に届かないため差し替えた。GPU 上の本番学習は未実行
- 出力: `BeatriceTraining/<名前>/<名前>_paraphernalia_<step>.zip`（Windows の beatrice-client 用）と `<名前>_checkpoint_<step>.pt.gz`（iPhone 版の Core ML 変換用）
