"""Evaluate tagging accuracy against the golden set.

Usage:
    python scripts/eval_accuracy.py [--api http://127.0.0.1:8000/api/v1] \
                                    [--out eval_reports/<name>.json]
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

SENSITIVE_SET = {
    "蘿莉", "正太", "嬰兒", "兒童",
    "強制", "強姦", "亂倫", "獵奇", "食人", "斷肢",
    "肛交", "觸手", "調教", "綁縛", "輪姦", "獸交",
    "中出", "顏射", "射精", "口交", "乳交", "手淫",
    "自慰", "足交", "戀足", "偷窺", "露出",
    "綠帽", "NTR", "少女", "少年", "偽娘",
}


def tag_image(api_base: str, image_path: Path) -> Dict[str, Any]:
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{api_base}/tag-cover",
            files={"file": (image_path.name, f, mime)},
            data={"top_k": "10", "confidence_threshold": "0.3"},
            timeout=240,
        )
    r.raise_for_status()
    return r.json()


def metrics(actual_tags: List[str], expected: Dict[str, List[str]]) -> Dict[str, float]:
    actual = set(actual_tags)
    must_have = set(expected.get("must_have", []))
    must_not = set(expected.get("must_not_have", []))
    nice = set(expected.get("nice_to_have", []))

    tp = len(actual & must_have)
    fp = len(actual & must_not)
    fn = len(must_have - actual)
    nice_hits = len(actual & nice)
    # Sensitive FP: substring-aware. Compound tags like "巨乳蘿莉" contain the
    # sensitive substring "蘿莉" and count as a leak even when the literal
    # compound isn't in must_not_have. Prevents the set-intersection escape
    # hatch where category-leak tags slip through a word-list blocklist.
    sensitive_fp = sum(
        1 for tag in actual if any(s in tag for s in SENSITIVE_SET)
    )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "nice_hits": nice_hits,
        # Flag for the summary aggregator: images with empty must_have
        # can't contribute meaningful precision/recall (tp is always 0 by
        # construction). Aggregation filters them out so acceptance gates
        # reflect only images where P/R is well-defined.
        "has_must_have": len(must_have) > 0,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "sensitive_fp": sensitive_fp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--out", default="eval_reports/eval.json")
    parser.add_argument("--golden", default="tests/golden/expected.json")
    parser.add_argument("--images-dir", default="tests/golden/images")
    args = parser.parse_args()

    expected_all = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    images_dir = Path(args.images_dir)

    per_image = []
    latencies = []
    for image_name, expected in expected_all.items():
        img_path = images_dir / image_name
        if not img_path.exists():
            print(f"SKIP {image_name}: not found", file=sys.stderr)
            continue
        t0 = time.time()
        try:
            response = tag_image(args.api, img_path)
        except Exception as e:
            print(f"FAIL {image_name}: {e}", file=sys.stderr)
            per_image.append({"image": image_name, "error": str(e)})
            continue
        latency = time.time() - t0
        latencies.append(latency)

        actual_tags = [t["tag"] for t in response.get("tags", [])]
        m = metrics(actual_tags, expected)
        m["latency_s"] = round(latency, 1)
        m["actual_tags"] = actual_tags
        m["image"] = image_name
        per_image.append(m)
        print(f"{image_name}  P={m['precision']} R={m['recall']} F1={m['f1']} "
              f"sens_fp={m['sensitive_fp']} {latency:.1f}s")

    scored = [m for m in per_image if "precision" in m]
    errored = [m for m in per_image if "error" in m]
    # Only images with a non-empty must_have can yield meaningful P/R —
    # without required tags, tp is always 0 and the gates are unreachable
    # by construction. Aggregate P/R over this subset only.
    pr_scored = [m for m in scored if m.get("has_must_have")]

    if scored:
        if pr_scored:
            mean_p = round(statistics.mean(m["precision"] for m in pr_scored), 3)
            mean_r = round(statistics.mean(m["recall"] for m in pr_scored), 3)
            mean_f = round(statistics.mean(m["f1"] for m in pr_scored), 3)
        else:
            mean_p = mean_r = mean_f = None  # no must_have anywhere
        summary = {
            "n_total": len(per_image),
            "n_scored": len(scored),
            "n_scored_with_must_have": len(pr_scored),
            "n_errored": len(errored),
            "median_latency_s": round(statistics.median(latencies), 1) if latencies else None,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
            "mean_f1": mean_f,
            "total_sensitive_fp": sum(m.get("sensitive_fp", 0) for m in scored),
            "sensitive_fp_per_scored_image": round(
                sum(m.get("sensitive_fp", 0) for m in scored) / len(scored), 3
            ),
        }
    else:
        summary = {
            "n_total": len(per_image),
            "n_scored": 0,
            "n_scored_with_must_have": 0,
            "n_errored": len(errored),
            "median_latency_s": None,
            "mean_precision": None,
            "mean_recall": None,
            "mean_f1": None,
            "total_sensitive_fp": 0,
            "sensitive_fp_per_scored_image": 0.0,
        }

    out = {"summary": summary, "per_image": per_image}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    if not scored:
        print(f"\nERROR: no scored rows ({len(errored)} errored). Eval failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
