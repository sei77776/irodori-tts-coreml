# Third-party notices and model provenance

This is a community Core ML conversion. The source-code license does not replace
the conditions of the checkpoint, tokenizer, audio codec or reference recordings.

| Component | Source and attribution | Declared license / included text |
|---|---|---|
| Irodori TTS v4.1 Small MF weights | [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF), revision `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e` | MIT, [text](LICENSES/MIT.txt) |
| Irodori implementation and derived conversion modules | [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS/tree/89f9d8fbd4d51ea019867ee1197725ede1df13c5), Copyright (c) 2026 Aratako | MIT, [text](LICENSES/MIT.txt) |
| Japanese ModernBERT encoder / tokenizer | [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m/tree/77675fc96a7e445e982e2ba90246b816efc74ec6), SB Intuitions and contributors | MIT, [text](LICENSES/ModernBERT-MIT.txt) |
| Japanese semantic DACVAE | [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim/tree/47376ee24834d7a05a48ebabfe3cde29b3c5e214), Aratako and contributors | Model card declares MIT; ancestry below still applies |
| AudioSeal implementation and watermark generator/detector weights | [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal), [facebook/audioseal](https://huggingface.co/facebook/audioseal/tree/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba); Copyright (c) Meta Platforms, Inc. and affiliates. | MIT, [text](LICENSES/AudioSeal-MIT.txt) |
| DACVAE implementation | [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae), Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. | Apache-2.0, [text](LICENSES/Apache-2.0.txt) |
| DACVAE base weights | [facebook/dacvae-watermarked](https://huggingface.co/facebook/dacvae-watermarked/tree/8680102d141858a21bd533543966a2eb2e569f92) | Apache-2.0, [Meta clarification for weights](https://huggingface.co/facebook/dacvae-watermarked/discussions/1); [text](LICENSES/Apache-2.0.txt) |
| Descript DAC architecture | [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec), Copyright (c) 2023-present, Descript | MIT, [text](LICENSES/Descript-MIT.txt) |
| ONNX intermediate conversion | [raratu/Onsei-iOS-Models](https://huggingface.co/raratu/Onsei-iOS-Models/tree/158b33fdb7fe95753922586c76bc8640feafd897) | Inherits the component terms above; immutable input digests are in `Conversion/sources.lock.json` in the code repository |

The runtime package contains this v4.1 ModernBERT tokenizer, not the llm-jp
tokenizer used by older Irodori conversions. No upstream Python environment,
Apple SDK, ONNX Runtime binary, user recording or voice-feature cache is bundled.

Changes: Core ML ML Programs replace runtime ONNX execution; DiT uses the existing
mixed-linear precision policy and four integration steps. Seven auxiliary models
use FP32. Decoder stages 1–3 use the existing FP16 policy with overlapping tiles;
stage 0 remains FP32. Reference statistics use the same Apple libc++ sampling
sequence. The conversion performs no additional quantization or retraining.

Meta organization member Matt Le (`lematt1991`) explicitly confirmed on
2025-12-19 that the DACVAE model weights are also Apache-2.0 in the official
[license discussion](https://huggingface.co/facebook/dacvae-watermarked/discussions/1).
The model README still contains a SAM sentence; that stale sentence is recorded
alongside this explicit weights clarification, not silently treated as corrected.
The license-review record is LICENSE_REVIEW.md in the model bundle and
`docs/LICENSE_REVIEW.md` in the code repository. Preserve Apache-2.0 for the
Meta-derived portions, MIT for the other listed portions, their attribution,
and notices of the Core ML conversion.

Follow the upstream [Irodori usage conditions and disclaimer](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF).
Use voices you are authorized to use; do not impersonate people without consent
or present synthetic speech as their authentic recording.
Samples identify their output as synthetic speech. Permission to run a model
does not establish permission to distribute another person's recordings.

Conversion-only dependencies are installed separately. Their licenses include
[coremltools BSD-3-Clause](https://github.com/apple/coremltools/blob/main/LICENSE.txt),
[onnx2torch Apache-2.0](https://github.com/ENOT-AutoDL/onnx2torch/blob/main/LICENSE)
and the respective PyTorch, NumPy, ONNX and Transformers licenses. Those tools
are not redistributed as vendored code or environments in this repository.

## Real-time voice changer (BeatriceVC, sample app)

| Component | Source and attribution | Declared license / included text |
|---|---|---|
| Beatrice v2 model definition and pretrained weights | [fierce-cats/beatrice-trainer](https://huggingface.co/fierce-cats/beatrice-trainer) 2.0.0-rc.0 | MIT |
| "標準" voices (`Examples/BeatriceAssets/`) | Speakers of the pretrained model, trained on LibriTTS-R and other corpora listed in beatrice-trainer `assets/README.md` | LibriTTS-R: CC BY 4.0 |
| "つくよみちゃん" voice (`Examples/BeatriceAssets/tsukuyomi/`) | Pretrained model fine-tuned on the [つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/) (CV.夢前黎). Required credit: 本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。■つくよみちゃんコーパス（CV.夢前黎） https://tyc.rei-yumesaki.net/material/corpus/ | Tsukuyomi-chan corpus terms (commercial use allowed with the credit above; prohibited uses of the output are listed in `Examples/BeatriceAssets/tsukuyomi/README.md` and in the app). No corpus audio is redistributed. |
