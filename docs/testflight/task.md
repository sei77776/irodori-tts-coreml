# TestFlight 配布 — タスク

- [x] `.github/workflows/testflight.yml` を作成
- [x] `Info.plist` に `ITSAppUsesNonExemptEncryption = NO` を追加
- [ ] Apple Developer で Bundle ID を登録し、App Store Connect にアプリを作成
- [ ] App Store Connect API キー（Admin）を発行し Secrets に登録
- [ ] リポジトリ変数 `APPLE_TEAM_ID` / `IOS_BUNDLE_ID` を登録
- [ ] Actions を有効化してワークフローを実行
- [ ] TestFlight にビルドが表示され、実機でインストールできることを確認
