# blue-tanuki-core 改修監査メモ

## 実施内容
- `App` を上流制御本体として再配線
  - デフォルトで `RuleDirector` を接続
  - built-in modules (`memory`, `file_ops`, `web_fetch`) を自動登録
  - pending を `FilePendingStore` へ変更し再起動耐性を付与
- optional 依存欠落で全体が起動不能になる欠陥を修正
  - `aiosqlite` 非依存で動く async sqlite shim を追加
  - `tenacity` 非依存の retry 実装へ変更
- ControlPlane の状態機械を補強
  - resume 時に `request_id` を新規発番
  - module dispatch の再実行防止
  - pre-LLM suspend/resume 時の二重 fold 防止
  - suspend 状態の保存順を修正し approval bypass を実効化
- 永続 pending 復元時の型崩れを修正
  - `Message` / `ModuleRequest` 復元
  - `plan_obj` の hydrate を追加

## 検証
- `pytest -q` -> 90 passed
- `python -m compileall -q src` -> success

## 残課題
- `web_fetch` は実ネットワーク環境依存のため、この環境では疎通確認を完遂していない
- `FilePendingStore` は JSON ファイル方式。多プロセス完全同期までは未対応
- `AnthropicBackend` は tenacity 依存を外したが、retry policy は最小実装
- README / DESIGN の文言は一部まだ旧記述が残る可能性あり
