# blue-tanuki-core

監査優先の **上流制御コア** です。役割は次の3つに限定します。

- inbound を gate で評価する
- director で module / llm に分岐する
- suspend / resume / audit を一貫した状態機械で保持する

下流 executor や UI は本リポジトリの責務ではありません。

## 含むもの

- `ControlPlane` — 5-hook gate と suspend / resume を持つ本体
- `App` — settings / audit / pending / pipe / director / modules の起動配線
- `RuleDirector` — 最小のルーティング規則
- built-in modules
  - `memory`
  - `file_ops`
  - `web_fetch`
- `AuditLog` — append-only JSONL + hash chain + verify
- `FilePendingStore` — restart-safe pending
- `LLMPipe` — backend failover / budget guard / prompt size guard
- `StubBackend` / `AnthropicBackend`
- CLI
  - `chat`
  - `verify`
  - `replay`
  - `audit`
  - `doctor`

## 含めないもの

- Telegram / Slack / GUI
- workflow engine
- multi-node coordination
- sealed evaluator logic

## 依存方針

最小依存で動くように整理済みです。

- 必須: `pydantic`, `pydantic-settings`, `anyio`, `httpx`, `pyyaml`
- 任意高速化: `aiosqlite` (`pip install .[speedups]`)

## quick start

```bash
pip install -e ".[dev]"
pytest -q

# stub backend
python -m blue_tanuki_core chat --stub

# anthropic backend
export ANTHROPIC_API_KEY=...
python -m blue_tanuki_core chat --anthropic
```

## ディレクトリ

```text
blue-tanuki-core/
├── README.md
├── policy.example.yaml
├── pyproject.toml
├── src/blue_tanuki_core/
│   ├── app.py
│   ├── audit.py
│   ├── control_plane.py
│   ├── gate.py
│   ├── pending.py
│   ├── pipe.py
│   ├── settings.py
│   ├── backend/
│   └── modules/
└── tests/
```

## 実装境界

- **core の責務**は「上流で止める・保留する・記録する」までです。
- 実業務の irreversible action は、別の operator layer または deterministic layer に委譲してください。

## companion

公開 API 殻は別リポジトリ `hds-public-shell` として分離しています。
`blue-tanuki-core` はローカル制御本体、`hds-public-shell` は外向け F→M→C 判断 API です。
