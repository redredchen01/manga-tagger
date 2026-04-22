"""Evaluate tagging accuracy against the golden set.

Usage:
    python scripts/eval_accuracy.py [--api http://127.0.0.1:8000/api/v1] \
                                    [--out eval_reports/<name>.json]
"""
from __future__ import annotations

import argparse
import json
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
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{api_base}/tag-cover",
            files={"file": (image_path.name, f, "image/jpeg")},
            data={"top_k": "10", "confidence_threshold": "0.3"},
            timeout=240,
        )
    r.raise_for_status()
    return r.json()


def metrics(actual_tags: List[str], expected: Dict[str, List[str]]) -> Dict[str, float]:
    actual = set(actual_tags)
    must_have = set(expected.get("must_have", []))
    must_not = set(expected.get("must_not_have", []))

    tp = len(actual & must_have)
    fp = len(actual & must_not)
    fn = len(must_have - actual)

    sensitive_actual = actual & SENSITIVE_SET
    sensitive_must_not = must_not & SENSITIVE_SET
    sensitive_fp = len(sensitive_actual & sensitive_must_not)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn,
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

    summary = {
        "n": len(per_image),
        "median_latency_s": round(statistics.median(latencies), 1) if latencies else None,
        "mean_precision": round(statistics.mean(m["precision"] for m in per_image if "precision" in m), 3) if per_image else 0,
        "mean_recall": round(statistics.mean(m["recall"] for m in per_image if "recall" in m), 3) if per_image else 0,
        "mean_f1": round(statistics.mean(m["f1"] for m in per_image if "f1" in m), 3) if per_image else 0,
        "total_sensitive_fp": sum(m.get("sensitive_fp", 0) for m in per_image),
        "sensitive_fp_per_image": round(sum(m.get("sensitive_fp", 0) for m in per_image) / max(len(per_image), 1), 3),
    }

    out = {"summary": summary, "per_image": per_image}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
