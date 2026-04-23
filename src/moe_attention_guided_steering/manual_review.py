from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Union

from .reference_data import ReferenceConceptSuite


QUESTION_PATTERN = re.compile(r'to the (?:question|request): "([^"]+)"')


@dataclass
class ManualReviewCase:
    """One three-way comparison to inspect by hand."""

    concept: str
    evaluation_version: int
    evaluation_question: str
    baseline_response: str = ""
    moesteer_response: str = ""
    attention_guided_moesteer_response: str = ""
    comparison_notes: str = ""
    preferred_condition: str = ""


@dataclass
class ManualReviewPlan:
    """A reproducible set of manual review cases for one concept family."""

    concept_type: str
    evaluation_family: str
    seed: int
    sampled_concepts: List[str]
    sampled_evaluation_versions: List[int]
    evaluation_questions_by_version: Dict[int, str]
    cases: List[ManualReviewCase]


def extract_evaluation_question(prompt_template: str) -> str:
    """Extract the model-facing question from an evaluation prompt template.

    The imported upstream prompt files are evaluator instructions. Each file
    contains one quoted question or request that should be asked to the model
    whose outputs we want to inspect. This helper pulls out that quoted string so
    we can build manual test sheets.

    Inputs:
    - `prompt_template`: full multi-line evaluator prompt from
      `data/evaluation_prompts/*.txt`.

    Returns:
    - the literal question/request string that should be posed to the model.
    """
    match = QUESTION_PATTERN.search(prompt_template)
    if match is None:
        raise ValueError("Could not find a quoted model-facing question in the evaluation prompt.")
    return match.group(1)


def build_manual_review_plan(
    concept_suite: ReferenceConceptSuite,
    concept_sample_size: int = 5,
    question_sample_size: int = 5,
    seed: int = 7,
) -> ManualReviewPlan:
    """Build a reproducible manual inspection grid for three output conditions.

    This utility is intentionally model-agnostic. It does not run generation.
    Instead, it selects a subset of concepts and evaluation questions so we can
    inspect three outputs by hand for each case:

    - baseline (no steering),
    - original MoESteer,
    - attention-guided MoESteer.

    Inputs:
    - `concept_suite`: one concept family plus its evaluation templates.
    - `concept_sample_size`: number of concepts to sample from the family.
    - `question_sample_size`: number of evaluation questions to sample.
    - `seed`: random seed used for reproducible sampling.

    Returns:
    - `ManualReviewPlan`: sampled concepts, sampled question versions, extracted
      model-facing questions, and the full concept-question cross product of
      review cases.
    """
    if concept_sample_size <= 0:
        raise ValueError("concept_sample_size must be positive.")
    if question_sample_size <= 0:
        raise ValueError("question_sample_size must be positive.")
    if concept_sample_size > len(concept_suite.concepts):
        raise ValueError("concept_sample_size exceeds the number of available concepts.")

    available_versions = sorted(concept_suite.evaluation_prompts_by_version)
    if question_sample_size > len(available_versions):
        raise ValueError("question_sample_size exceeds the number of available evaluation prompts.")

    rng = random.Random(seed)
    sampled_concepts = rng.sample(concept_suite.concepts, concept_sample_size)
    sampled_versions = rng.sample(available_versions, question_sample_size)

    evaluation_questions_by_version = {
        version: extract_evaluation_question(concept_suite.evaluation_prompts_by_version[version])
        for version in sampled_versions
    }

    cases = [
        ManualReviewCase(
            concept=concept,
            evaluation_version=version,
            evaluation_question=evaluation_questions_by_version[version],
        )
        for concept in sampled_concepts
        for version in sampled_versions
    ]

    return ManualReviewPlan(
        concept_type=concept_suite.concept_type,
        evaluation_family=concept_suite.evaluation_family,
        seed=seed,
        sampled_concepts=sampled_concepts,
        sampled_evaluation_versions=sampled_versions,
        evaluation_questions_by_version=evaluation_questions_by_version,
        cases=cases,
    )


def manual_review_plan_to_dict(plan: ManualReviewPlan) -> Dict[str, Any]:
    """Serialize a manual review plan into a JSON-friendly dictionary."""
    return {
        "concept_type": plan.concept_type,
        "evaluation_family": plan.evaluation_family,
        "seed": plan.seed,
        "sampled_concepts": list(plan.sampled_concepts),
        "sampled_evaluation_versions": list(plan.sampled_evaluation_versions),
        "evaluation_questions_by_version": dict(plan.evaluation_questions_by_version),
        "cases": [asdict(case) for case in plan.cases],
    }


def manual_review_plan_from_dict(data: Dict[str, Any]) -> ManualReviewPlan:
    """Rehydrate a `ManualReviewPlan` from JSON-friendly serialized data."""
    return ManualReviewPlan(
        concept_type=data["concept_type"],
        evaluation_family=data["evaluation_family"],
        seed=int(data["seed"]),
        sampled_concepts=list(data["sampled_concepts"]),
        sampled_evaluation_versions=[int(version) for version in data["sampled_evaluation_versions"]],
        evaluation_questions_by_version={
            int(version): question
            for version, question in data["evaluation_questions_by_version"].items()
        },
        cases=[ManualReviewCase(**case) for case in data["cases"]],
    )


def load_manual_review_plan(path: Union[str, Path]) -> ManualReviewPlan:
    """Load a serialized manual review plan from disk."""
    return manual_review_plan_from_dict(json.loads(Path(path).read_text()))


def build_manual_review_markdown(plan: ManualReviewPlan) -> str:
    """Render a collaborator-friendly Markdown review sheet or filled report.

    If response fields are still blank, this reads like a worksheet. If the plan
    has already been populated with model generations, the same renderer becomes a
    lightweight qualitative report.
    """
    def display_response(value: str) -> str:
        return value if value.strip() else "Pending generation"

    lines = [
        "# Manual Fear Review",
        "",
        f"- Concept family: **{plan.concept_type}**",
        f"- Evaluation family: **{plan.evaluation_family}**",
        f"- Sampling seed: **{plan.seed}**",
        f"- Number of sampled concepts: **{len(plan.sampled_concepts)}**",
        f"- Number of sampled questions: **{len(plan.sampled_evaluation_versions)}**",
        f"- Number of three-condition comparisons: **{len(plan.cases)}**",
        "",
        "## Sampled Concepts",
        "",
    ]

    for concept in plan.sampled_concepts:
        lines.append(f"- {concept}")

    lines.extend(["", "## Sampled Evaluation Questions", ""])
    for version in plan.sampled_evaluation_versions:
        lines.append(f"- v{version}: {plan.evaluation_questions_by_version[version]}")

    for concept in plan.sampled_concepts:
        lines.extend(["", f"## {concept}", ""])
        for version in plan.sampled_evaluation_versions:
            matching_case = next(
                case
                for case in plan.cases
                if case.concept == concept and case.evaluation_version == version
            )
            question = matching_case.evaluation_question
            lines.extend(
                [
                    f"### Eval v{version}",
                    "",
                    f"- Question: {question}",
                    "- Baseline response:",
                    f"  {display_response(matching_case.baseline_response)}",
                    "- Original MoESteer response:",
                    f"  {display_response(matching_case.moesteer_response)}",
                    "- Attention-guided MoESteer response:",
                    f"  {display_response(matching_case.attention_guided_moesteer_response)}",
                    "- Comparison notes:",
                    f"  {matching_case.comparison_notes.strip() or 'Pending annotation'}",
                    "- Preferred condition:",
                    f"  {matching_case.preferred_condition.strip() or 'Not chosen yet'}",
                    "",
                ]
            )

    return "\n".join(lines)


def build_manual_review_html(
    plan: ManualReviewPlan,
    title: str = "Qualitative Review",
    companion_attention_report: str = "",
) -> str:
    """Render a browsable HTML report for qualitative response inspection.

    This report is intentionally centered on the manual comparison task rather
    than the raw attention numbers. If response fields are still blank, the HTML
    makes that explicit so the next missing step is obvious.
    """
    def display_response(value: str) -> str:
        return html.escape(value) if value.strip() else "<span class='pending'>Pending generation</span>"

    completed_cases = sum(
        int(
            bool(case.baseline_response.strip())
            or bool(case.moesteer_response.strip())
            or bool(case.attention_guided_moesteer_response.strip())
        )
        for case in plan.cases
    )

    summary_rows = []
    for concept in plan.sampled_concepts:
        concept_cases = [case for case in plan.cases if case.concept == concept]
        completed = sum(
            int(
                bool(case.baseline_response.strip())
                or bool(case.moesteer_response.strip())
                or bool(case.attention_guided_moesteer_response.strip())
            )
            for case in concept_cases
        )
        summary_rows.append(
            f"<tr><td><strong>{html.escape(concept)}</strong></td><td>{completed}/{len(concept_cases)}</td></tr>"
        )
    summary_rows.append(
        f"<tr style='border-top:2px solid #333'><td><strong>Overall</strong></td><td><strong>{completed_cases}/{len(plan.cases)}</strong></td></tr>"
    )

    concept_sections: List[str] = []
    for concept in plan.sampled_concepts:
        concept_cases = [case for case in plan.cases if case.concept == concept]
        for rank, case in enumerate(concept_cases, start=1):
            moesteer_filled = bool(case.moesteer_response.strip())
            attention_filled = bool(case.attention_guided_moesteer_response.strip())
            moesteer_badge = (
                "<span class='badge b1'>Filled</span>"
                if moesteer_filled
                else "<span class='badge b0'>Pending</span>"
            )
            attention_badge = (
                "<span class='badge b1'>Filled</span>"
                if attention_filled
                else "<span class='badge b0'>Pending</span>"
            )
            comparison_note = (
                html.escape(case.comparison_notes)
                if case.comparison_notes.strip()
                else "<span class='pending'>Pending annotation</span>"
            )
            preferred_condition = (
                html.escape(case.preferred_condition)
                if case.preferred_condition.strip()
                else "<span class='pending'>Not chosen yet</span>"
            )

            concept_sections.append(
                f"""
                <section>
                <h2>#{rank} &mdash; <strong>{html.escape(concept)}</strong> &middot; prompt v{case.evaluation_version}</h2>
                <p><strong>Prompt:</strong> {html.escape(case.evaluation_question)}</p>
                <p class='lbl lbl-bl'>Baseline (no steering):</p>
                <pre>{display_response(case.baseline_response)}</pre>
                <div class='cols'>
                  <div class='col col-ms'>
                    <p class='lbl lbl-ms'>Original MoESteer {moesteer_badge}</p>
                    <pre>{display_response(case.moesteer_response)}</pre>
                  </div>
                  <div class='col col-ag'>
                    <p class='lbl lbl-ag'>Attention-guided MoESteer {attention_badge}</p>
                    <pre>{display_response(case.attention_guided_moesteer_response)}</pre>
                  </div>
                </div>
                <div class='note'>
                  <strong>Comparison notes:</strong> {comparison_note}<br>
                  <strong>Preferred condition:</strong> {preferred_condition}
                </div>
                </section>
                """
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5;color:#24292f;padding:0 1rem;}}
h1{{font-size:1.45rem;}} h2{{font-size:1.05rem;margin:2rem 0 .5rem;}}
section{{border:1px solid #d0d7de;border-radius:8px;padding:1rem;margin-bottom:1.5rem;background:#fafafa;}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}}
.col{{min-width:0;}}
.col-ms{{border-left:4px solid #0550ae;padding-left:.75rem;}}
.col-ag{{border-left:4px solid #cf222e;padding-left:.75rem;}}
.lbl{{font-weight:700;font-size:.9rem;margin-bottom:.3rem;}}
.lbl-ms{{color:#0550ae;}} .lbl-ag{{color:#cf222e;}} .lbl-bl{{color:#555;}}
pre{{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;padding:.6rem;border-radius:6px;font-size:.85rem;margin:.3rem 0 .6rem;min-height:4.5rem;}}
.badge{{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-weight:600;font-size:.82rem;margin-left:.3rem;}}
.b1{{background:#d4edda;color:#155724;}} .b0{{background:#f8d7da;color:#721c24;}}
.meta{{color:#57606a;font-size:.85rem;}}
table{{border-collapse:collapse;margin:1rem 0;}}
td,th{{padding:.4rem .8rem;border:1px solid #d0d7de;text-align:left;}}
th{{background:#f6f8fa;}}
.note{{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:.6rem 1rem;margin:1rem 0;font-size:.9rem;}}
.pending{{color:#8a5a00;font-style:italic;}}
code{{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;}}
@media (max-width: 820px){{.cols{{grid-template-columns:1fr;}}}}
  </style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class='note'>
<strong>What are we comparing?</strong>
<ul style='margin:.4rem 0 .2rem;padding-left:1.3rem;'>
<li><span style='color:#555;font-weight:700;'>Baseline</span> &mdash; the model with no steering intervention at all.</li>
<li><span style='color:#0550ae;font-weight:700;'>Original MoESteer</span> &mdash; the future MoE steering condition using the original token-choice strategy.</li>
<li><span style='color:#cf222e;font-weight:700;'>Attention-guided MoESteer</span> &mdash; the future MoE steering condition using the attention-selected token positions from this repo.</li>
</ul>
<strong>How to use this page:</strong> compare the three answers for each (fear, question) pair, then annotate which steered version feels more faithful to the target fear without becoming incoherent.
</div>
<p class='meta'>Concept family: <code>{html.escape(plan.concept_type)}</code> &middot; Evaluation family: <code>{html.escape(plan.evaluation_family)}</code> &middot; Seed: <code>{plan.seed}</code> &middot; Companion attention report: <code>{html.escape(companion_attention_report) if companion_attention_report else 'not linked'}</code></p>
<h2>Summary</h2>
<table><tr><th>Concept</th><th>Cases with any filled response</th></tr>
{''.join(summary_rows)}
</table>
{''.join(concept_sections)}
</body>
</html>
"""
