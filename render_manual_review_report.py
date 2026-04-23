#!/usr/bin/env python3
"""Render a browsable qualitative review report from a manual review plan JSON."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.manual_review import (  # noqa: E402
    build_manual_review_html,
    build_manual_review_markdown,
    load_manual_review_plan,
)


def main() -> None:
    """Read a manual review plan JSON and write HTML + Markdown reports."""
    parser = argparse.ArgumentParser(
        description="Render a qualitative review report from a manual review plan."
    )
    parser.add_argument(
        "--plan-json",
        default="outputs/manual_fear_review/manual_review_plan.json",
        help="Path to the manual review plan JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manual_fear_review",
        help="Directory where the rendered review files should be written.",
    )
    parser.add_argument(
        "--title",
        default="Qualitative Fear Comparison",
        help="Title shown in the rendered report.",
    )
    parser.add_argument(
        "--attention-report",
        default="",
        help="Optional companion attention report path to display in the HTML header.",
    )
    args = parser.parse_args()

    plan = load_manual_review_plan(args.plan_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "qualitative_review.md"
    html_path = output_dir / "qualitative_review.html"

    markdown_path.write_text(build_manual_review_markdown(plan))
    html_path.write_text(
        build_manual_review_html(
            plan,
            title=args.title,
            companion_report=args.attention_report,
        )
    )

    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
