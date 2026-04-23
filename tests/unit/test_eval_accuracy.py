"""Unit tests for the metrics() function in scripts/eval_accuracy.py.

Pure-function tests — no network, no I/O. Loads the module by file path so
we don't need scripts/ to be a package.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load scripts/eval_accuracy.py as a module without requiring it to be a package
_spec = importlib.util.spec_from_file_location(
    "eval_accuracy", Path(__file__).parent.parent.parent / "scripts" / "eval_accuracy.py"
)
eval_accuracy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_accuracy)


def test_perfect_match():
    expected = {"must_have": ["a", "b"], "must_not_have": ["x"], "nice_to_have": []}
    m = eval_accuracy.metrics(["a", "b"], expected)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["sensitive_fp"] == 0


def test_partial_recall():
    expected = {"must_have": ["a", "b"], "must_not_have": [], "nice_to_have": []}
    m = eval_accuracy.metrics(["a"], expected)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5


def test_false_positive_counts():
    expected = {"must_have": ["a"], "must_not_have": ["bad"], "nice_to_have": []}
    m = eval_accuracy.metrics(["a", "bad", "neutral"], expected)
    assert m["tp"] == 1
    assert m["fp"] == 1  # "bad" is in must_not_have; "neutral" is neither
    assert m["fn"] == 0
    assert m["sensitive_fp"] == 0  # "bad" not in SENSITIVE_SET


def test_sensitive_fp_only_counts_blocklist_intersection():
    # 蘿莉 is in SENSITIVE_SET; "bad" is not
    expected = {"must_have": [], "must_not_have": ["蘿莉", "bad"], "nice_to_have": []}
    m = eval_accuracy.metrics(["蘿莉", "bad"], expected)
    assert m["fp"] == 2
    assert m["sensitive_fp"] == 1  # only 蘿莉 counts


def test_sensitive_fp_caught_even_if_not_in_must_not_have():
    # sensitive_fp is now substring-checked against SENSITIVE_SET directly,
    # so a 蘿莉 leak is flagged even when the golden file's must_not_have
    # list is empty. This prevents the "implementer forgot to widen the
    # blocklist" footgun from hiding category-level leaks.
    expected = {"must_have": [], "must_not_have": [], "nice_to_have": []}
    m = eval_accuracy.metrics(["蘿莉"], expected)
    assert m["fp"] == 0  # must_not_have is still empty, so per-image fp stays 0
    assert m["sensitive_fp"] == 1  # but SENSITIVE_SET membership catches it


def test_nice_to_have_hits_counted():
    expected = {"must_have": ["a"], "must_not_have": [], "nice_to_have": ["b", "c"]}
    m = eval_accuracy.metrics(["a", "b", "c", "d"], expected)
    assert m["nice_hits"] == 2  # b and c
    assert m["tp"] == 1
    assert m["fp"] == 0  # d is neither must_have nor must_not_have


def test_empty_inputs_no_crash():
    expected = {"must_have": [], "must_not_have": [], "nice_to_have": []}
    m = eval_accuracy.metrics([], expected)
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_missing_keys_default_safely():
    # expected dict only has must_have; metrics shouldn't KeyError
    m = eval_accuracy.metrics(["a"], {"must_have": ["a"]})
    assert m["tp"] == 1
    assert m["fp"] == 0


def test_sensitive_fp_substring_catches_compound_tags():
    """巨乳蘿莉 contains 蘿莉 (in SENSITIVE_SET) — must be counted as
    a sensitive leak even though it's not literally in must_not_have.
    Prior behavior used strict set intersection and missed this."""
    expected = {"must_have": [], "must_not_have": ["蘿莉"], "nice_to_have": []}
    m = eval_accuracy.metrics(["巨乳蘿莉"], expected)
    assert m["sensitive_fp"] == 1


def test_sensitive_fp_substring_catches_multiple_sensitive_terms():
    expected = {"must_have": [], "must_not_have": [], "nice_to_have": []}
    m = eval_accuracy.metrics(["低含量蘿莉", "輪姦party", "safe_tag"], expected)
    # 低含量蘿莉 contains 蘿莉; 輪姦party contains 輪姦; safe_tag contains neither.
    assert m["sensitive_fp"] == 2


def test_has_must_have_flag_true():
    expected = {"must_have": ["a"], "must_not_have": [], "nice_to_have": []}
    m = eval_accuracy.metrics([], expected)
    assert m["has_must_have"] is True


def test_has_must_have_flag_false():
    expected = {"must_have": [], "must_not_have": ["x"], "nice_to_have": ["y"]}
    m = eval_accuracy.metrics(["y"], expected)
    assert m["has_must_have"] is False
