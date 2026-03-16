#!/usr/bin/env python3
"""
Script to standardize the first 100 core tags from the tag library.
"""

import sys

sys.path.insert(0, ".")

from app.services.tag_description_standardizer import TagDescriptionStandardizer
import json
from pathlib import Path


def main():
    print("Starting Tag Library Standardization")
    print("=" * 60)

    # Initialize standardizer
    standardizer = TagDescriptionStandardizer()

    # Process first 100 tags
    input_path = "51標籤庫.json"
    output_path = "data/tags_enhanced_top100.json"

    print(f"\nLoading tags from: {input_path}")
    print(f"Will process first 100 tags")
    print(f"Output will be saved to: {output_path}")

    # Process
    stats = standardizer.process_tag_library(
        input_path=input_path, output_path=output_path, limit=100
    )

    print("\n" + "=" * 60)
    print("Standardization Complete!")
    print(f"\nStatistics:")
    print(f"   Total processed: {stats['total']}")
    print(f"   By category: {dict(stats['by_category'])}")
    print(f"   Avg description length: {stats['avg_description_length']:.1f} chars")
    print(f"   Avg visual keywords: {stats['avg_keywords']:.1f}")

    print(f"\nEnhanced tags saved to: {output_path}")

    # Show sample
    print("\nSample enhanced tags:")
    with open(output_path, "r", encoding="utf-8") as f:
        enhanced = json.load(f)
        for tag in enhanced[:3]:
            print(f"\n   Tag: {tag['tag_name']}")
            print(f"      Category: {tag['category']}")
            print(f"      Keywords: {', '.join(tag['visual_keywords'][:5])}")
            print(f"      Confidence: {tag['required_confidence']}")


if __name__ == "__main__":
    main()
