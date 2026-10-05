# TestFlight 配布 — 手順

## 1. Apple 側の準備（初回のみ）
1. [Identifiers](https://developer.apple.com/account/resources/identifiers/list) で App ID を登録（例: `com.yourname.irodori`）。Capabilities の追加は不要
2. [App Store Connect](https://appstoreconnect.apple.com/apps) →「新規App」→ 上の Bundle ID を選んで作成
3. ユーザとアクセス → 統合 → App Store Connect API →「チーム キー」を **Admin** 権限で発行し、`.p8` をダウンロード（再ダウンロード不可）。Key ID と Issuer ID を控える
4. [Membership](https://developer.apple.com/account#MembershipDetailsCard) で Team ID（10 文字）を確認

## 2. GitHub 側の設定（Settings → Secrets and variables → Actions）
| 種類 | 名前 | 値 |
|---|---|---|
| Secret | `ASC_KEY_ID` | API キーの Key ID |
| Secret | `ASC_ISSUER_ID` | Issuer ID |
| Secret | `ASC_KEY_P8` | `.p8` ファイルの中身（`-----BEGIN PRIVATE KEY-----` から末尾まで） |
| Variable | `APPLE_TEAM_ID` | Team ID |
| Variable | `IOS_BUNDLE_ID` | 1 で登録した Bundle ID |

フォークしたリポジトリは Actions が無効なので、Actions タブで有効化する。

## 3. 実行
Actions → TestFlight → Run workflow（`claude/testflight` への push でも起動）。
成功すると 10〜30 分ほどで App Store Connect の TestFlight タブにビルドが現れる。内部テスターに自分を追加し、iPhone の TestFlight アプリからインストールする。

## 4. アプリの使い方
モデル（約 2.99 GB）は同梱されていない。アプリの「モデル」→「URLからダウンロード」に `docs/HUGGINGFACE.md` の manifest URL を入力して取得する。端末に十分な空き容量が必要。

## よくある失敗
- `No Accounts` / `No signing certificate`: API キーが Admin でない
- `No suitable application records were found`: App Store Connect にアプリを作っていない、または Bundle ID 不一致
- `The bundle version must be higher`: 同じビルド番号が既にある。もう一度実行すれば run_number が増える
