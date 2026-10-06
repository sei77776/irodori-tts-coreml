# Beatrice v2 Core ML ベンチ

Irodori とは独立した実験用ディレクトリ。public リポジトリの無料 macOS ランナーで測るためにここに置いている。

- 対象: [fierce-cats/beatrice-trainer](https://huggingface.co/fierce-cats/beatrice-trainer)（MIT）のモデル定義と事前学習重みを、非公開の推論ライブラリを使わずに Core ML 化したもの
- 測るもの: Core ML と PyTorch の数値一致、compute unit ごとの 1 回あたり推論時間、ランナーの機種
- 実行: `.github/workflows/beatrice-coreml-bench.yml`（このディレクトリへの push か手動実行）。結果は artifact `beatrice-coreml-bench` と Step Summary
- 経緯と PoC 全体: sei77776/claude-code の `BeatriceVC/poc/`（private）
