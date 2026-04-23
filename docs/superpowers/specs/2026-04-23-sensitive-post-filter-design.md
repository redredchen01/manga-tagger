# Sensitive Substring Post-Filter — Design Spec

**Date:** 2026-04-23
**Status:** Approved (brainstorming sign-off)
**Scope:** Phase 1 follow-up — close the substring-leak vector that lets compound
sensitive tags (e.g. `巨乳蘿莉`) bypass Stage 10 verification.

## 1. Problem

`recommender._verify_and_calibrate` (Stage 10) gates sensitive tags with
`is_sensitive = rec.tag in settings.SENSITIVE_TAGS` — an **exact** set
membership check. Compound tags that *contain* a sensitive substring but are
not themselves in the set (e.g. `巨乳蘿莉` contains `蘿莉`) skip the gate
entirely and reach the final output.

The eval harness (`scripts/eval_accuracy.py:57`) uses substring matching, so it
correctly counts these compounds as sensitive false positives. The pipeline
does not. The asymmetry produced an observed
`sensitive_fp_per_scored_image = 0.333` on the 3-image starter golden set
(memory: `sensitive_substring_leak`), failing spec §6.2's `≤ 0.3` target.

## 2. Goals & Non-Goals

**Goals:**
- Eliminate the substring-leak path: any tag containing a `SENSITIVE_TAGS`
  substring but not exactly in the set must be visually re-verified or dropped.
- Keep the existing exact-match path (verify-or-penalize) unchanged.
- Stay strictly within Stage 10; no prompt or eval changes.
- Provide a rollback flag.

**Non-Goals:**
- Not raising the bar for exact-match singletons (e.g. plain `蘿莉`) — out of
  scope; current `verify_sensitive_tag` + ×0.7 penalty stays.
- Not changing `SENSITIVE_TAGS` membership.
- Not adding new VLM prompts, new models, or new pipeline stages.
- Not making the `≤ 0.3` acceptance gate pass on the starter set — that
  requires Workstream V (golden-set expansion). This work only proves the
  leak vector is closed.

## 3. Decisions (Brainstorming Outcomes)

| Question | Decision | Rationale |
|---|---|---|
| Behavior strength | **B**: drop substring sensitives unless verified | Hard gate on the leak vector; A's ×0.7 penalty doesn't actually drop. |
| What counts as verified | **R2**: `verify_sensitive_tag` returns true | Visually grounded; RAG-based fallback (R1) is too loose for compounds. |
| Affect exact-match path? | **(a)** No — current behavior preserved | Scope discipline: fix what leaked, not what didn't. |

## 4. Architecture

Single insertion point: `_verify_and_calibrate` in
`app/domain/tag/recommender.py`. The new substring branch sits **next to**
the existing exact-match branch and **before** frequency calibration / RAG
support boost / mutual exclusivity. Drops short-circuit `continue` so dropped
recs incur no further calibration cost.

### Decision flow per recommendation

```
exact match in SENSITIVE_TAGS?
├─ yes → existing path (verify-or-penalize, unchanged)
└─ no  → SUBSTRING_FILTER_ENABLED && any(s in tag for s in SENSITIVE_TAGS)?
         ├─ no  → normal path (no sensitive treatment)
         └─ yes → vlm_service && image_bytes?
                  ├─ no  → drop + WARN
                  └─ yes → verify_sensitive_tag(image, tag)
                           ├─ True  → keep, append " | substring-verified" to reason
                           ├─ False → drop + WARN
                           └─ raise → drop + WARN (fail-closed)
```

## 5. Components

### 5.1 `app/domain/tag/recommender.py`

Replace lines ~580–594 with classify→gate structure:

```python
exact_sensitive = rec.tag in settings.SENSITIVE_TAGS
substring_sensitive = (
    not exact_sensitive
    and settings.SENSITIVE_SUBSTRING_FILTER_ENABLED
    and any(s in rec.tag for s in settings.SENSITIVE_TAGS)
)

if exact_sensitive:
    # unchanged: verify-or-penalize
    ...
elif substring_sensitive:
    if not (vlm_service and image_bytes):
        logger.warning("substring-sensitive drop (no verifier): %s", rec.tag)
        continue
    try:
        ok = await vlm_service.verify_sensitive_tag(image_bytes, rec.tag)
    except Exception as e:
        logger.warning("substring-sensitive drop (verify error %s): %s",
                       type(e).__name__, rec.tag)
        continue
    if not ok:
        logger.warning("substring-sensitive drop (verify failed): %s", rec.tag)
        continue
    rec.reason += " | substring-verified"

verified.append(rec)
```

### 5.2 `app/core/config.py`

Add to `Settings`:

```python
SENSITIVE_SUBSTRING_FILTER_ENABLED: bool = True
```

No env-var binding needed; rollback is a one-line default flip.

### 5.3 `tests/unit/test_sensitive_substring_filter.py` (new)

Five cases, all on a fake `vlm_service` with controllable `verify_sensitive_tag`:

| # | Setup | Expected |
|---|---|---|
| 1 | compound (`巨乳蘿莉`) + image_bytes + verify→True | tag retained, reason contains `substring-verified` |
| 2 | compound + image_bytes + verify→False | tag dropped |
| 3 | compound + no image_bytes | tag dropped |
| 4 | exact (`蘿莉`) + no image_bytes | tag retained with ×0.7 penalty (regression guard for decision (a)) |
| 5 | compound + flag disabled | tag retained (rollback path) |

## 6. Data Flow

```
VLM JSON tags ─┐
RAG matches  ──┼─→ Stages 1–9 (unchanged)
desc rescue  ──┘                  │
                                   ↓
                  Stage 10 _verify_and_calibrate
                                   │
                          per rec: classify
                                   │
                  ┌────────────────┼────────────────┐
                  │ exact          │ substring      │ neither
                  │ (unchanged)    │ (NEW gate)     │ (passthrough)
                  └────────────────┴────────────────┘
                                   ↓
                  freq calib / RAG support / mutex (unchanged)
                                   ↓
                              top_k output
```

Substring drops occur **before** calibration, so dropped recs never participate
in downstream boosting or mutex resolution.

## 7. Error Handling

- `verify_sensitive_tag` raises → fail-closed (drop). Sensitive content
  defaults to refusal under uncertainty.
- `image_bytes is None` while `vlm_service` is set → treat as "no verifier" → drop.
- Empty `SENSITIVE_TAGS` → `any(...)` short-circuits to False; substring branch
  never fires; degrades to current behavior.

## 8. Logging

- All drops: `logger.warning` with format
  `substring-sensitive drop (<reason>): <tag>`,
  `<reason> ∈ {no verifier, verify failed, verify error <ExcType>}`.
- Keeps: no extra log; `rec.reason` carries `" | substring-verified"` so
  eval/inspection sees it.

## 9. Testing Strategy

- **Unit:** 5 cases above. No integration tests in CI — real VLM is ~90s/call
  per memory `vlm_qwen_behavior`.
- **Manual smoke:** run `scripts/eval_accuracy.py` on the starter golden set
  after merge. Acceptance signal:
  `sensitive_fp_per_scored_image < 0.333` (strictly below the pre-fix value).
  This proves the leak vector is closed; it does **not** claim the spec §6.2
  `≤ 0.3` gate is met (that requires golden-set expansion).

## 10. Rollback

Set `SENSITIVE_SUBSTRING_FILTER_ENABLED = False` in `app/core/config.py`
defaults. Substring branch becomes inert; exact-match path continues unchanged.
No data migration, no schema change.

## 11. Out of Scope (deferred)

- Golden set expansion (Workstream V) — required for acceptance-gate evidence.
- Two-stage VLM spec (Workstream W) — orthogonal latency/quality work.
- Tightening exact-match singletons to drop-without-image — possible follow-up
  if eval shows leakage there too.
