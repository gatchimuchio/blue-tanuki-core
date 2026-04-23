# PATCH AUDIT — policy hardening / upstream-control runtime default

Date: 2026-04-20
Target: `blue-tanuki-core`

## Objective

Replace the permissive `AllowAll` runtime default with a shipped safe-default policy, and make cost / side-effect guardrails active in the upstream control plane without requiring manual wiring.

## Implemented

1. **Shipped default runtime policy**
   - Added `src/blue_tanuki_core/policy.default.yaml`
   - Used automatically when `Settings.policy_path` is not set and no gate override is injected

2. **Runtime default changed from permissive to policy-driven**
   - `App.start()` now loads the bundled default policy instead of falling back to `AllowAllGate`

3. **Pre-LLM payload sizing made observable to gate**
   - Added `payload_size` to inbound / pre_llm / post_llm gate subjects so payload-size-based rules are enforceable

4. **Packaging updated**
   - Added hatch wheel force-include rule for the bundled policy file

5. **Regression / behavior tests added**
   - default policy suspends file write
   - default policy suspends external fetch
   - default policy stops sensitive credential-like file writes
   - default policy suspends oversized LLM payloads

## Default policy behavior

### Suspends for approval
- `file_ops.write`
- `file_ops.append`
- `file_ops.delete`
- `web_fetch.get`
- pre-LLM payloads above configured proxy threshold
- high-cost model name patterns (`opus`, `gpt-5-pro`, `o1-pro`, `ultra`)

### Stops
- credential-like file mutations (`.env*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_*`)
- git metadata mutation / deletion (`.git*` family)
- provider-filtered LLM output
- post-module side effects above cap

### Clarifies
- empty inbound input

## Validation

- `pytest -q` → passed
- `python -m compileall -q src` → passed

## Residual limitations

1. Policy is still regex / rule driven, not semantic-risk scored.
2. High-cost detection is currently proxy-based (`model_match`, `payload_size_gt`), not real token-cost accounting.
3. Approval subject payloads are generic and could be enriched further for operator UX.
4. Default thresholds are static constants; no per-tenant / per-environment policy overlay yet.

## Result

The core now behaves like an actual upstream control layer by default:
- no silent mutating file operation,
- no silent network egress,
- no silent oversized LLM blast radius,
- no permissive startup due to missing explicit policy configuration.
