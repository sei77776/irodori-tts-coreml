# TestFlight 配布 — 実装計画

## 目的
`Examples/IrodoriSamples.xcodeproj` の `IrodoriiOS` を GitHub Actions の macOS ランナーでアーカイブし、TestFlight へアップロードする。

## 方針
- ランナー: `macos-26`（App Store Connect は最新 iOS SDK でのビルドを要求する）
- 署名: Xcode の自動署名（クラウド管理証明書）を App Store Connect API キーで行う。証明書 (.p12) とプロファイルの手動管理は不要
- Bundle ID: リポジトリ変数 `IOS_BUNDLE_ID` で `org.example.irodori.coreml.ios` をビルド時に置換（リポジトリには個人の ID を書かない）
- ビルド番号: `github.run_number`
- 輸出規制: `ITSAppUsesNonExemptEncryption = NO`（HTTPS と CryptoKit のハッシュのみで免除対象）

## 却下案
- 手動署名（p12 + プロファイルを Secrets に格納）: 確実だが証明書・プロファイルの更新作業が発生する。自動署名が失敗した場合のフォールバックとする
- EAS / fastlane: Expo ではなく、追加の依存を入れる必要がない
