# SENSITIVE_TAGS Fix & Golden Set Expansion — Design

**Date**: 2026-04-23  
**Scope**: Fix str-vs-set asymmetry in recommender.py:581, add production-format tests, initiate golden set expansion (10-image batch)

---

## Overview

Three interconnected work items to resolve lingering issues from the substring-sensitive post-filter fix (commits b77390b..313994a):

1. **Code fix**: SENSITIVE_TAGS str-vs-set asymmetry (recommender.py:581)
2. **Test coverage**: Production-shaped string fixture + parameterized tests
3. **Golden set**: First batch of 10 curated images with real `must_have` values

Implementation approach: **Method 3 (sequential PRs)**
- PR 1: Code fix + tests (immediate)
- PR 2: Data (golden set expansion, follow-up)

---

## 1. SENSITIVE_TAGS Str-vs-Set Asymmetry

### Problem

**recommender.py:581** uses `settings.SENSITIVE_TAGS` (a comma-separated string):
```python
exact_sensitive = rec.tag in settings.SENSITIVE_TAGS
```

**recommender.py:585** uses `settings.sensitive_tags` (a set):
```python
and any(s in rec.tag for s in settings.sensitive_tags)
```

**Root cause**: 
- Line 581 was added before the `@computed_field` property `sensitive_tags` was introduced
- The `in` operator on a string does substring matching, not set membership
- Tests mock both as sets, so the divergence is invisible in CI

**Impact**: 
- If SENSITIVE_TAGS contains "蘿莉" and a tag is "蘿莉色" (contains the substring), line 581 would incorrectly match it as an exact-sensitive tag
- This could cause verification paths to execute when they shouldn't, or confidence penalties to apply incorrectly

### Solution

**Single-line change** (recommender.py:581):
```python
# Before
exact_sensitive = rec.tag in settings.SENSITIVE_TAGS

# After
exact_sensitive = rec.tag in settings.sensitive_tags
```

**Why this approach**:
- Minimal, focused fix
- Uses the proper set type (computed from string at instantiation)
- Matches the pattern already in use at line 585
- No refactoring needed elsewhere

### Test Strategy

**New test**: `test_sensitive_tags_with_string_format`
- Patches `settings.SENSITIVE_TAGS` as a production comma-separated string (e.g., `"蘿莉,正太,嬰兒"`)
- Patches `settings.sensitive_tags` as the corresponding set
- Verifies that `exact_sensitive` and `substring_sensitive` logic works correctly with both formats
- Catches the string-vs-set bug that unit tests normally hide

**Parameterized existing tests**:
- Extend current sensitive-tag tests in test_tag_recommender.py to run with:
  1. Settings mocked as sets (current behavior)
  2. Settings with SENSITIVE_TAGS as string, sensitive_tags as set (production format)
- Ensures behavior is consistent regardless of mock format

**Scope**: 
- Only touch test_tag_recommender.py and recommender.py
- No changes to config.py or other modules

---

## 2. Golden Set Expansion (Batch 1: 10 Images)

### Problem

Current golden set (tests/golden/expected.json):
- Only 4 images
- **All have empty `must_have` arrays**
- Precision/recall metrics are mathematically unreachable: if `must_have = []`, then `tp = 0` always
- Spec §6.2 acceptance gate (≤ 0.3) cannot be measured reliably

Memory note: "golden set has empty must_have — starter expected.json makes P/R gates mathematically unreachable"

### Solution: Phased Expansion

**Batch 1 (this PR series)**: Add 10 curated, high-quality images
**Future (Workstream V)**: Expand to 30+ images for full acceptance gate validation

Each image in Batch 1 must have:
```json
{
  "description": "detailed visual description",
  "must_have": ["tag1", "tag2"],        // tags that should confidently appear
  "must_not_have": [敏感标签列表],        // safety check
  "nice_to_have": ["optional_tag"]      // aspirational matches
}
```

### Image Curation Criteria

- **Diversity**: Multiple styles (character-focused, scene-heavy, close-up, full-body, etc.)
- **Clarity**: Each image should have obvious tags that the 611-tag library can match
- **Complexity range**: Mix of simple (1–2 tags) and complex (5+ tags) scenarios
- **Sensitivity check**: All must_not_have lists include the standard sensitive set
- **No duplicates**: Visually distinct from the 4 existing golden images

### Impact

Once Batch 1 is in place:
- Eval reports will show `n_scored_with_must_have > 0`
- Precision/recall metrics become meaningful
- Foundation for Workstream V (full 30+ image set)

### Scope

- Add 10 JPEG/PNG image files to tests/golden/images/
- Extend tests/golden/expected.json with 10 new entries
- No changes to eval_accuracy.py or scoring logic
- No changes to recommender or config modules

---

## 3. Implementation Strategy: Sequential PRs

### PR 1 (Code): SENSITIVE_TAGS Fix + Tests
**Deliverable**: 
- recommender.py:581 fixed
- test_tag_recommender.py: new production-format test + parameterized existing tests
- All tests pass

**Timeline**: 1–2 hours

**Review focus**: 
- Correctness of the str-to-set change
- Test coverage (does the new test catch the bug?)
- No regressions in other recommender logic

---

### PR 2 (Data): Golden Set Batch 1
**Deliverable**:
- 10 curated images in tests/golden/images/
- Extended tests/golden/expected.json
- Eval report showing n_scored_with_must_have = 10 (or close)

**Timeline**: Parallel to PR 1; can be prepared while PR 1 is under review

**Review focus**:
- Image diversity and clarity
- Correctness of must_have/must_not_have annotations
- Eval metrics confirm meaningful P/R computation

---

## 4. Why Separate PRs?

**Decoupling concerns**:
- Code fix is urgent (bug fix)
- Image curation is time-consuming (manual work)
- Separating allows fast code iteration + parallel data collection

**Clear ownership**:
- PR 1: Technical correctness
- PR 2: Data quality and diversity

**Future-proof**:
- Once Batch 1 works, Workstream V (30+ images) can be planned with confidence
- Each data expansion is a separate PR, easy to iterate

---

## Success Criteria

✓ recommender.py:581 uses `settings.sensitive_tags` (set), not `settings.SENSITIVE_TAGS` (string)  
✓ New production-format test added and passing  
✓ Parameterized tests validate both string and set mock formats  
✓ All existing recommender tests still pass  
✓ PR 1 merged and deployed  
✓ 10 golden images added to tests/golden/images/  
✓ tests/golden/expected.json extended with 10 entries (each with non-empty must_have)  
✓ Eval report: n_scored_with_must_have = 10 and meaningful P/R metrics computed  
✓ PR 2 merged and ready for Workstream V planning

---

## Open Questions / Future Work

- Workstream V planning: How to prioritize the next 20+ images? (Image sources, tagging workflow, review process)
- Spec §6.2 acceptance gate: Once Batch 1 is stable, determine if current implementation meets ≤ 0.3 threshold or if further tuning is needed
- SENSITIVE_SUBSTRING_FILTER_ENABLED flag: Monitor edge cases in production; may need future refinement based on real-world data

