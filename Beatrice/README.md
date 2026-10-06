# Beatrice リアルタイム変声（実験）

サンプルアプリの「リアルタイム変声（実験）」画面で使う、Beatrice v2 の Core ML 版と話者データ。

- モデル定義と事前学習重み: [fierce-cats/beatrice-trainer](https://huggingface.co/fierce-cats/beatrice-trainer)（2.0.0-rc.0、MIT）。非公開の推論ライブラリは使っていない
- 同梱話者: 事前学習モデルに含まれる話者（LibriTTS-R 由来、CC BY 4.0）から、埋め込みが互いに遠い 5 名を番号で表示
- 方式: 疑似ストリーミング。毎チャンク、窓 = 左文脈 + チャンク + 先読み 40 ms をモデル全体に通し、チャンク分だけ出力する。パルス位相を引き継ぎ、20 ms クロスフェードする

## 構成

| 場所 | 中身 |
|---|---|
| `Examples/BeatriceAssets/` | アプリに同梱するフォルダ（フォルダ参照）。`BeatriceFP32.mlpackage`（全 FP32）、`BeatriceMixed.mlpackage`（FP16 + ピッチ系・VQ を FP32）、`voices.json`、`voice_XXX.bin`、`common.bin` |
| `Sources/BeatriceVC/` | Swift 実装。DSP（パルス overlap-add・雑音・768 点 post filter、vDSP）、窓処理、Core ML 呼び出し、AVAudioEngine 入出力 |
| `Examples/Shared/BeatriceView.swift` | 画面 |
| `Tests/BeatriceVCTests/` | Python 参照出力（`Golden/`）との照合テスト。`swift test` で実行 |
| `Beatrice/tools/` | 変換・書き出しスクリプト |

モデルは Core ML の複数関数モデル（iOS 18 / macOS 15 以降）で、窓長ごとに関数 `w64`（左文脈約 0.5 秒）、`w116`（約 1 秒）、`w216`（約 2 秒）を持つ。重みは関数間で共有される。

## 話者・モデルの作り直し

```sh
cd Beatrice/tools
python -m pip install -r requirements.txt
bash fetch_assets.sh                     # beatrice-trainer と事前学習モデルを ./work に取得
python -I export_ios_assets.py --assets ../../Examples/BeatriceAssets --golden ../../Tests/BeatriceVCTests/Golden
```

Colab などで学習した話者を使う場合は、学習済みチェックポイント（`checkpoint_latest.pt.gz` など、`net_g` を含むもの）を
`BEATRICE_NET_G` で指定し、`--speakers all` で書き出す。ファインチューニングでボコーダの重みも変わるため、
モデルと話者データは必ず一緒に作り直す。

```sh
BEATRICE_NET_G=/path/to/checkpoint_latest.pt.gz python -I export_ios_assets.py \
    --assets ../../Examples/BeatriceAssets --speakers all --label "声"
```

`voice_XXX.bin` は float32 リトルエンディアンで、話者埋め込み 256、KV 384×128、VQ コードブック 512×128 をこの順に並べたもの。
`common.bin` は IR 窓 512 とフォルマントシフト埋め込み 9×256。
