#!/usr/bin/env python3
"""
Script to validate tag library quality and generate report.
"""

import sys

sys.path.insert(0, ".")

from app.services.tag_quality_validator import TagQualityValidator


def main():
    print("Tag Library Quality Validation")
    print("=" * 60)

    # Validate the enhanced tags
    validator = TagQualityValidator()

    input_path = "data/tags_enhanced_top100.json"
    output_path = "reports/tag_quality_report.txt"

    print(f"\nValidating: {input_path}")
    print(f"Report will be saved to: {output_path}")

    # Generate report
    report_text = validator.generate_report(input_path, output_path)

    # Print report
    print("\n" + report_text)

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
