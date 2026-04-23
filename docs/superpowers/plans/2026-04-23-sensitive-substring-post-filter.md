# Sensitive Substring Post-Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the substring-leak vector that lets compound sensitive tags like `巨乳蘿莉` bypass Stage 10 verification, by adding a drop-unless-`verify_sensitive_tag`-passes branch alongside the existing exact-match path.

**Architecture:** Single insertion point inside `_verify_and_calibrate` in `app/domain/tag/recommender.py`. A new `elif substring_sensitive` branch sits next to the existing `if exact_sensitive` branch and runs **before** frequency calibration / RAG support boost / mutex resolution, so dropped recs cost nothing downstream. Controlled by a new `SENSITIVE_SUBSTRING_FILTER_ENABLED` boolean flag (default `True`) for one-line rollback.

**Tech Stack:** Python 3, pydantic-settings, pytest (+pytest-asyncio), unittest.mock.

**Spec reference:** `docs/superpowers/specs/2026-04-23-sensitive-post-filter-design.md`

---

## Pre-implementation Notes (read first)

**Quirk you need to know:** `settings.SENSITIVE_TAGS` (uppercase) is a **string** in production — `pydantic` stores it as a comma-joined env-overridable Field (`app/core/config.py:170`). `settings.sensitive_tags` (lowercase) is a **set** computed from that string (`app/core/config.py:175-178`). The existing exact-match check at `app/domain/tag/recommender.py:581` uses the uppercase string and works only by accident of substring-on-flat-string semantics.

**For this work:**
- The existing exact-match line stays as-is (spec scope: don't change it).
- The new substring branch MUST iterate over the proper set. Use `settings.sensitive_tags` (lowercase).
- Mock both `mock.SENSITIVE_TAGS = {"蘿莉"}` and `mock.sensitive_tags = {"蘿莉"}` in new tests so both branches behave deterministically. The existing test fixture (`tests/unit/test_tag_recommender.py:38`) already mocks the uppercase as a set, so this aligns.

---

## Task 1: Add `SENSITIVE_SUBSTRING_FILTER_ENABLED` config flag

**Files:**
- Modify: `app/core/config.py` (add field to `Settings` class)
- Test: `tests/unit/test_config.py`

- [ ] **Step 1.1: Write the failing test**

Append to `tests/unit/test_config.py` (inside the existing `TestSettings` class — find it by searching for `class TestSettings` or `def test_sensitive_tags_default`):

```python
    def test_sensitive_substring_filter_enabled_default(self):
        """The substring post-filter flag defaults to True."""
        from app.core.config import Settings
        s = Settings()
        assert s.SENSITIVE_SUBSTRING_FILTER_ENABLED is True
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_config.py::TestSettings::test_sensitive_substring_filter_enabled_default -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'SENSITIVE_SUBSTRING_FILTER_ENABLED'`.

- [ ] **Step 1.3: Add the flag**

In `app/core/config.py`, immediately after the `sensitive_tags` computed_field block (after the closing `}` on line ~178), insert:

```python
    # Drop substring-sensitive compound tags (e.g. 巨乳蘿莉) unless
    # verify_sensitive_tag confirms. Rollback by flipping default to False.
    SENSITIVE_SUBSTRING_FILTER_ENABLED: bool = True
```

- [ ] **Step 1.4: Run the test to verify it passes**

Run: `pytest tests/unit/test_config.py::TestSettings::test_sensitive_substring_filter_enabled_default -v`
Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add app/core/config.py tests/unit/test_config.py
git commit -m "config: add SENSITIVE_SUBSTRING_FILTER_ENABLED flag (default True)"
```

---

## Task 2: Write the failing unit tests for the substring gate

**Files:**
- Create: `tests/unit/test_sensitive_substring_filter.py`

- [ ] **Step 2.1: Create the test file with all 5 cases**

Create `tests/unit/test_sensitive_substring_filter.py` with this content:

```python
"""Unit tests for the substring-sensitive post-filter in Stage 10.

Covers spec docs/superpowers/specs/2026-04-23-sensitive-post-filter-design.md:
B (drop unless verified) + R2 (verify_sensitive_tag is the only rescue) +
(a) exact-match path unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommendation, TagRecommenderService


@pytest.fixture
def patched_settings():
    """Mock settings with a non-empty sensitive set containing only `蘿莉`.

    Both uppercase (used by the existing exact branch via `in`) and lowercase
    (used by the new substring branch via iteration) are populated.
    """
    with patch("app.domain.tag.recommender.settings") as mock:
        mock.SENSITIVE_TAGS = {"蘿莉"}
        mock.sensitive_tags = {"蘿莉"}
        mock.SENSITIVE_SUBSTRING_FILTER_ENABLED = True
        # Calibration / boost knobs - neutralize so we can assert on raw confidence
        mock.TAG_FREQUENCY_CALIBRATION = {}
        mock.EXACT_MATCH_PENALTY = {}
        mock.VISUAL_FEATURE_BOOST = {}
        mock.MIN_ACCEPTABLE_CONFIDENCE = 0.0
        mock.RAG_SUPPORT_BOOST = 1.0
        mock.RAG_SUPPORT_DECAY = 1.0
        mock.SEMANTIC_SIBLINGS = {}
        mock.MUTUAL_EXCLUSIVITY = {}
        mock.TAG_HIERARCHY = {}
        yield mock


def _make_rec(tag: str, confidence: float = 0.8, source: str = "vlm_json"):
    return TagRecommendation(
        tag=tag, confidence=confidence, source=source, reason="test"
    )


@pytest.mark.asyncio
async def test_compound_with_image_and_verify_true_is_kept(patched_settings):
    """巨乳蘿莉 + image_bytes + verify→True → retained, reason marks substring-verified."""
    svc = TagRecommenderService()
    vlm_service = MagicMock()
    vlm_service.verify_sensitive_tag = AsyncMock(return_value=True)

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=vlm_service,
        image_bytes=b"fake-image-bytes",
        rag_matches=[],
        vlm_analysis={},
    )

    assert any(r.tag == "巨乳蘿莉" for r in out)
    kept = next(r for r in out if r.tag == "巨乳蘿莉")
    assert "substring-verified" in kept.reason
    vlm_service.verify_sensitive_tag.assert_awaited_once_with(
        b"fake-image-bytes", "巨乳蘿莉"
    )


@pytest.mark.asyncio
async def test_compound_with_image_and_verify_false_is_dropped(patched_settings):
    """巨乳蘿莉 + image_bytes + verify→False → dropped."""
    svc = TagRecommenderService()
    vlm_service = MagicMock()
    vlm_service.verify_sensitive_tag = AsyncMock(return_value=False)

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=vlm_service,
        image_bytes=b"fake-image-bytes",
        rag_matches=[],
        vlm_analysis={},
    )

    assert all(r.tag != "巨乳蘿莉" for r in out)


@pytest.mark.asyncio
async def test_compound_without_image_bytes_is_dropped(patched_settings):
    """巨乳蘿莉 + no image_bytes → dropped (no rescue path available)."""
    svc = TagRecommenderService()

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=None,
        image_bytes=None,
        rag_matches=[],
        vlm_analysis={},
    )

    assert all(r.tag != "巨乳蘿莉" for r in out)


@pytest.mark.asyncio
async def test_exact_singleton_without_image_bytes_is_kept_with_penalty(
    patched_settings,
):
    """蘿莉 (exact match) + no image_bytes → kept, penalized ×0.7 (decision (a))."""
    svc = TagRecommenderService()

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("蘿莉", confidence=0.8)],
        vlm_service=None,
        image_bytes=None,
        rag_matches=[],
        vlm_analysis={},
    )

    kept = next((r for r in out if r.tag == "蘿莉"), None)
    assert kept is not None
    # 0.8 * 0.7 = 0.56; safe_confidence rounds, so allow tolerance
    assert 0.55 <= kept.confidence <= 0.57
    assert "Unverified" in kept.reason


@pytest.mark.asyncio
async def test_compound_with_flag_disabled_is_kept(patched_settings):
    """Rollback path: flag=False → 巨乳蘿莉 retained even without verifier."""
    patched_settings.SENSITIVE_SUBSTRING_FILTER_ENABLED = False
    svc = TagRecommenderService()

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=None,
        image_bytes=None,
        rag_matches=[],
        vlm_analysis={},
    )

    assert any(r.tag == "巨乳蘿莉" for r in out)
```

- [ ] **Step 2.2: Run the new tests to verify they fail**

Run: `pytest tests/unit/test_sensitive_substring_filter.py -v`
Expected: tests 1, 2, 3, 5 should FAIL (compound currently slips through unchanged because `巨乳蘿莉 not in {"蘿莉"}` and there is no substring branch yet). Test 4 (exact singleton) should PASS — that's the regression baseline confirming we won't break decision (a).

Note: if all 5 tests pass already, something is wrong — stop and re-check; you probably patched the wrong module.

- [ ] **Step 2.3: Commit the failing tests**

```bash
git add tests/unit/test_sensitive_substring_filter.py
git commit -m "test: add failing cases for substring-sensitive post-filter"
```

---

## Task 3: Implement the substring gate in `_verify_and_calibrate`

**Files:**
- Modify: `app/domain/tag/recommender.py:578-594` (the per-rec sensitive block inside `_verify_and_calibrate`)

- [ ] **Step 3.1: Locate the existing block**

Open `app/domain/tag/recommender.py` and find the `_verify_and_calibrate` method (around line 568). The block to replace is the loop body that starts at line 580 and runs through line 594. For reference, the **current** code is:

```python
        for rec in recommendations:
            is_sensitive = rec.tag in settings.SENSITIVE_TAGS

            if is_sensitive and vlm_service and image_bytes:
                is_verified = await vlm_service.verify_sensitive_tag(image_bytes, rec.tag)
                if not is_verified:
                    logger.warning(f"Sensitive tag '{rec.tag}' failed verification, removing")
                    continue
                rec.reason += " | Verified"

            elif is_sensitive and not (vlm_service and image_bytes):
                rec.confidence = safe_confidence(rec.confidence * 0.7)
                rec.reason += " | Unverified (penalized)"

            verified.append(rec)
```

- [ ] **Step 3.2: Replace with the classify→gate structure**

Use Edit to replace the exact block above (`for rec in recommendations:` through the `verified.append(rec)` line — only the loop body for the existing sensitive logic) with:

```python
        for rec in recommendations:
            exact_sensitive = rec.tag in settings.SENSITIVE_TAGS
            substring_sensitive = (
                not exact_sensitive
                and settings.SENSITIVE_SUBSTRING_FILTER_ENABLED
                and any(s in rec.tag for s in settings.sensitive_tags)
            )

            if exact_sensitive and vlm_service and image_bytes:
                is_verified = await vlm_service.verify_sensitive_tag(image_bytes, rec.tag)
                if not is_verified:
                    logger.warning(f"Sensitive tag '{rec.tag}' failed verification, removing")
                    continue
                rec.reason += " | Verified"

            elif exact_sensitive and not (vlm_service and image_bytes):
                rec.confidence = safe_confidence(rec.confidence * 0.7)
                rec.reason += " | Unverified (penalized)"

            elif substring_sensitive:
                if not (vlm_service and image_bytes):
                    logger.warning(
                        "substring-sensitive drop (no verifier): %s", rec.tag
                    )
                    continue
                try:
                    ok = await vlm_service.verify_sensitive_tag(image_bytes, rec.tag)
                except Exception as e:
                    logger.warning(
                        "substring-sensitive drop (verify error %s): %s",
                        type(e).__name__,
                        rec.tag,
                    )
                    continue
                if not ok:
                    logger.warning(
                        "substring-sensitive drop (verify failed): %s", rec.tag
                    )
                    continue
                rec.reason += " | substring-verified"

            verified.append(rec)
```

- [ ] **Step 3.3: Run the new tests**

Run: `pytest tests/unit/test_sensitive_substring_filter.py -v`
Expected: all 5 PASS.

If any fail, do NOT modify the test to make it pass — re-read the implementation and confirm:
- `settings.sensitive_tags` (lowercase) is iterated in the substring check
- `settings.SENSITIVE_SUBSTRING_FILTER_ENABLED` short-circuits before iteration
- the new branch is `elif substring_sensitive`, not nested inside the exact branches
- `continue` (not `break`) is used for drops

- [ ] **Step 3.4: Run the full test suite to check for regressions**

Run: `pytest tests/unit/ -v`
Expected: all tests PASS. Pay special attention to `tests/unit/test_tag_recommender.py` — its existing fixture uses `mock.SENSITIVE_TAGS = set()` (empty), so neither branch will fire and behavior is unchanged.

If a non-related test fails, stop and investigate before continuing.

- [ ] **Step 3.5: Commit**

```bash
git add app/domain/tag/recommender.py
git commit -m "$(cat <<'EOF'
fix(recommender): substring sensitive post-filter (drop unless verified)

Closes the leak where compound tags like 巨乳蘿莉 bypass Stage 10 because
they are not exact members of SENSITIVE_TAGS. New elif substring_sensitive
branch drops the tag unless verify_sensitive_tag returns true; exact-match
path is unchanged. Gated by SENSITIVE_SUBSTRING_FILTER_ENABLED.

Spec: docs/superpowers/specs/2026-04-23-sensitive-post-filter-design.md
EOF
)"
```

---

## Task 4: Manual eval smoke verification

**Files:**
- No code changes. This task captures evidence for the verification gate in spec §9.

- [ ] **Step 4.1: Capture pre-fix baseline (skip if already on record)**

The baseline is documented in memory `sensitive_substring_leak.md` as `sensitive_fp_per_scored_image = 0.333` on the starter golden set under commit `e7b8490`. No need to re-run; cite this value.

- [ ] **Step 4.2: Run eval against the now-patched pipeline**

Run: `python scripts/eval_accuracy.py`
Expected output (in the summary line): `sensitive_fp_per_scored_image` strictly less than `0.333`. Most likely `0.000` because the only known leaker (`巨乳蘿莉` on `test_anime_detailed.jpg`) will now be dropped at the no-verifier branch (the eval script does not pass `image_bytes` to verify, so the substring branch will drop unconditionally).

If the value is **not** below 0.333, stop. The most likely cause is the eval invoking the recommender with a `vlm_service` that returns `True` from `verify_sensitive_tag`. Inspect `scripts/eval_accuracy.py` to see how it calls `recommend_tags` and whether `image_bytes`/`vlm_service` get passed through.

- [ ] **Step 4.3: Record the result**

Append a one-line note to the spec doc footer (if not already added by the executor):

```markdown
## Manual Verification (2026-04-23)

- Pre-fix `sensitive_fp_per_scored_image`: 0.333 (memory: sensitive_substring_leak)
- Post-fix `sensitive_fp_per_scored_image`: <value from Step 4.2>
- Verdict: leak vector closed (strictly below pre-fix value).
```

- [ ] **Step 4.4: Commit the evidence note**

```bash
git add docs/superpowers/specs/2026-04-23-sensitive-post-filter-design.md
git commit -m "docs: record sensitive post-filter manual eval result"
```

---

## Self-Review (executed during writing-plans, recorded for reader transparency)

**Spec coverage:**
- §3 decisions B/R2/(a): all three baked into Task 3's branch shape (drop unless `verify_sensitive_tag`; substring scope only; exact path untouched). ✓
- §5.1 code snippet: matches Task 3 step 3.2 verbatim. ✓
- §5.2 config flag: Task 1. ✓
- §5.3 test cases (5): Task 2 step 2.1 lists all 5 with same setup matrix. ✓
- §7 error handling (verify exception → drop): Task 3 step 3.2 wraps `verify_sensitive_tag` in `try/except Exception`. ✓
- §8 logging format: Task 3 step 3.2 uses the spec's exact `<reason>` strings. ✓
- §9 manual eval signal (strictly below 0.333): Task 4. ✓
- §10 rollback (flip default): Task 1 enables it; no separate task needed. ✓

**Placeholder scan:** No TBDs, no "add error handling later", no "similar to Task N", no missing code blocks. ✓

**Type consistency:** `settings.SENSITIVE_TAGS` (str/set in fixture) used only for `in` membership; `settings.sensitive_tags` (set) used only for iteration. `verify_sensitive_tag` signature `(image_bytes, tag) -> bool` consistent across implementation, tests, and spec. ✓
