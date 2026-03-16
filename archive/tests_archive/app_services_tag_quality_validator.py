"""
Tag Quality Validator
Validates tag quality and generates quality reports.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a quality issue with a tag."""

    tag_name: str
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str


@dataclass
class QualityReport:
    """Quality report for the tag library."""

    total_tags: int
    issues: List[QualityIssue] = field(default_factory=list)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    avg_description_length: float = 0.0
    tags_with_keywords: int = 0
    tags_with_conflicts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tags": self.total_tags,
            "issues_count": len(self.issues),
            "issues_by_severity": self._count_by_severity(),
            "category_distribution": self.category_distribution,
            "avg_description_length": self.avg_description_length,
            "tags_with_keywords": self.tags_with_keywords,
            "tags_with_conflicts": self.tags_with_conflicts,
            "issues": [
                {
                    "tag": issue.tag_name,
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
        }

    def _count_by_severity(self) -> Dict[str, int]:
        counts = Counter(issue.severity for issue in self.issues)
        return dict(counts)


class TagQualityValidator:
    """
    Validates tag quality and identifies issues.
    """

    def __init__(self):
        """Initialize the validator."""
        self.issues = []
        self.min_description_length = 10
        self.max_description_length = 500
        self.min_keywords = 2

    def validate_tag_library(self, tags_data: List[Dict[str, Any]]) -> QualityReport:
        """
        Validate entire tag library.

        Args:
            tags_data: List of tag dictionaries

        Returns:
            QualityReport with all issues found
        """
        self.issues = []

        logger.info(f"Validating {len(tags_data)} tags...")

        # Validate each tag
        for tag in tags_data:
            self._validate_single_tag(tag)

        # Check for duplicates
        self._check_duplicates(tags_data)

        # Calculate statistics
        category_dist = Counter(tag.get("category", "unknown") for tag in tags_data)

        desc_lengths = []
        tags_with_kw = 0
        tags_with_conf = 0

        for tag in tags_data:
            desc = tag.get("description", "")
            desc_lengths.append(len(desc))

            if tag.get("visual_keywords"):
                tags_with_kw += 1

            if tag.get("conflicting_tags"):
                tags_with_conf += 1

        avg_desc_len = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0

        report = QualityReport(
            total_tags=len(tags_data),
            issues=self.issues,
            category_distribution=dict(category_dist),
            avg_description_length=avg_desc_len,
            tags_with_keywords=tags_with_kw,
            tags_with_conflicts=tags_with_conf,
        )

        return report

    def _validate_single_tag(self, tag: Dict[str, Any]) -> None:
        """Validate a single tag."""
        tag_name = tag.get("tag_name", "")

        # Check 1: Tag name exists
        if not tag_name:
            self.issues.append(
                QualityIssue(
                    tag_name="UNKNOWN",
                    issue_type="missing_name",
                    severity="error",
                    message="Tag has no name",
                    suggestion="Add a tag_name field",
                )
            )
            return

        # Check 2: Description exists and has minimum length
        description = tag.get("description", "")
        if not description:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="missing_description",
                    severity="error",
                    message="Tag has no description",
                    suggestion=f"Add a description of at least {self.min_description_length} characters",
                )
            )
        elif len(description) < self.min_description_length:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="short_description",
                    severity="warning",
                    message=f"Description is too short ({len(description)} chars)",
                    suggestion=f"Extend description to at least {self.min_description_length} characters",
                )
            )
        elif len(description) > self.max_description_length:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="long_description",
                    severity="info",
                    message=f"Description is very long ({len(description)} chars)",
                    suggestion="Consider making description more concise",
                )
            )

        # Check 3: Visual keywords
        keywords = tag.get("visual_keywords", [])
        if not keywords:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="no_keywords",
                    severity="warning",
                    message="Tag has no visual keywords",
                    suggestion=f"Add at least {self.min_keywords} visual keywords for better matching",
                )
            )
        elif len(keywords) < self.min_keywords:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="few_keywords",
                    severity="info",
                    message=f"Tag has only {len(keywords)} keywords",
                    suggestion=f"Consider adding more keywords (aim for {self.min_keywords}+)",
                )
            )

        # Check 4: Category
        category = tag.get("category", "")
        if not category or category == "other":
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="uncategorized",
                    severity="info",
                    message="Tag is uncategorized or in 'other' category",
                    suggestion="Assign to appropriate category (character, body, clothing, action, theme)",
                )
            )

        # Check 5: Confidence threshold
        confidence = tag.get("required_confidence")
        if confidence is None:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="no_confidence",
                    severity="warning",
                    message="No required confidence specified",
                    suggestion="Set required_confidence (default: 0.65)",
                )
            )
        elif confidence < 0.5:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="low_confidence",
                    severity="info",
                    message=f"Very low confidence threshold ({confidence})",
                    suggestion="Consider raising confidence to at least 0.5",
                )
            )
        elif confidence > 0.9:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="high_confidence",
                    severity="info",
                    message=f"Very high confidence threshold ({confidence})",
                    suggestion="High confidence may reduce recall",
                )
            )

        # Check 6: Related tags consistency
        related = tag.get("related_tags", [])
        if len(related) > 10:
            self.issues.append(
                QualityIssue(
                    tag_name=tag_name,
                    issue_type="too_many_related",
                    severity="info",
                    message=f"Tag has {len(related)} related tags",
                    suggestion="Consider limiting to most important 5-10 related tags",
                )
            )

    def _check_duplicates(self, tags_data: List[Dict[str, Any]]) -> None:
        """Check for duplicate tag names."""
        names = [tag.get("tag_name", "") for tag in tags_data]
        duplicates = [name for name, count in Counter(names).items() if count > 1]

        for dup in duplicates:
            self.issues.append(
                QualityIssue(
                    tag_name=dup,
                    issue_type="duplicate",
                    severity="error",
                    message=f"Duplicate tag name found ({Counter(names)[dup]} times)",
                    suggestion="Remove duplicate entries",
                )
            )

    def generate_report(self, tags_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate and save quality report.

        Args:
            tags_path: Path to tags JSON file
            output_path: Optional path to save report

        Returns:
            Formatted report string
        """
        # Load tags
        with open(tags_path, "r", encoding="utf-8") as f:
            tags = json.load(f)

        # Validate
        report = self.validate_tag_library(tags)

        # Generate formatted report
        lines = []
        lines.append("=" * 60)
        lines.append("TAG LIBRARY QUALITY REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Total Tags: {report.total_tags}")
        lines.append(f"Total Issues: {len(report.issues)}")
        lines.append("")

        # Issues by severity
        lines.append("Issues by Severity:")
        severity_counts = report._count_by_severity()
        for sev in ["error", "warning", "info"]:
            count = severity_counts.get(sev, 0)
            lines.append(f"  {sev.upper()}: {count}")
        lines.append("")

        # Category distribution
        lines.append("Category Distribution:")
        for cat, count in sorted(report.category_distribution.items()):
            lines.append(f"  {cat}: {count}")
        lines.append("")

        # Statistics
        lines.append("Statistics:")
        lines.append(
            f"  Average Description Length: {report.avg_description_length:.1f} chars"
        )
        lines.append(
            f"  Tags with Keywords: {report.tags_with_keywords} ({report.tags_with_keywords / report.total_tags * 100:.1f}%)"
        )
        lines.append(f"  Tags with Conflicts: {report.tags_with_conflicts}")
        lines.append("")

        # Issues list
        if report.issues:
            lines.append("Issues Found:")
            lines.append("-" * 60)

            # Group by severity
            for severity in ["error", "warning", "info"]:
                sev_issues = [i for i in report.issues if i.severity == severity]
                if sev_issues:
                    lines.append(f"\n{severity.upper()} ({len(sev_issues)}):")
                    for issue in sev_issues[:10]:  # Show first 10 of each severity
                        lines.append(f"  [{issue.issue_type}] {issue.tag_name}")
                        lines.append(f"    {issue.message}")
                        lines.append(f"    Suggestion: {issue.suggestion}")

                    if len(sev_issues) > 10:
                        lines.append(f"    ... and {len(sev_issues) - 10} more")

        lines.append("")
        lines.append("=" * 60)

        report_text = "\n".join(lines)

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)

            # Also save JSON version
            json_path = str(Path(output_path).with_suffix(".json"))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Report saved to {output_path} and {json_path}")

        return report_text


if __name__ == "__main__":
    # Test the validator
    print("Testing Tag Quality Validator")
    print("=" * 60)

    validator = TagQualityValidator()

    # Test with sample data
    test_tags = [
        {
            "tag_name": "Test Tag 1",
            "description": "Short",
            "visual_keywords": [],
            "category": "other",
        },
        {
            "tag_name": "Test Tag 2",
            "description": "This is a proper description with enough length to pass validation",
            "visual_keywords": ["keyword1", "keyword2", "keyword3"],
            "category": "character",
            "required_confidence": 0.75,
        },
        {"tag_name": "", "description": "Missing name"},
    ]

    report = validator.validate_tag_library(test_tags)

    print(f"\nTotal tags: {report.total_tags}")
    print(f"Issues found: {len(report.issues)}")
    print(f"\nIssues by severity: {report._count_by_severity()}")

    if report.issues:
        print("\nFirst 5 issues:")
        for issue in report.issues[:5]:
            print(f"  [{issue.severity}] {issue.tag_name}: {issue.message}")

    print("\n" + "=" * 60)
    print("Test completed!")
