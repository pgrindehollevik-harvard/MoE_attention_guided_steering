from dataclasses import dataclass, field
import html
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Union

from .reference_data import ReferenceConceptSuite
from .upstream_prompt_datasets import build_concept_conditioned_evaluation_prompt


QUESTION_PATTERN = re.compile(r'to the (?:question|request): "([^"]+)"')

DEFAULT_CONDITION_ORDER = ["baseline", "steermoe"]
DEFAULT_CONDITION_LABELS = {
    "baseline": "OLMoE baseline (question only)",
    "steermoe": "OLMoE + SteerMoE (question only)",
    "attention_guided_moesteer": "Attention-guided MoESteer",
    "llama_3_1_8b_baseline": "Llama 3.1 8B baseline (question only)",
    "llama_3_1_8b_reference": "Llama 3.1 8B reference (question only)",
}


@dataclass
class ManualReviewCase:
    """One qualitative comparison case for a concept-question pair.

    Inputs and meaning:
    - `concept`: the sampled concept under evaluation, such as `Bugs`.
    - `evaluation_version`: upstream evaluation prompt version id.
    - `evaluation_question`: literal model-facing question asked at generation
      time.
    - `full_prompt_text`: exact prompt shown to the model for this review case.
      For steering evaluation this intentionally equals `evaluation_question`
      so the concept prefix is omitted at test time.
    - `prefix_conditioned_prompt_text`: diagnostic prompt that includes the
      upstream concept prefix. This is useful for model-suitability comparisons,
      but it is not the steering-test prompt.
    - `responses`: mapping from condition key to generated text. Example keys:
      - `baseline`
      - `steermoe`
      - `attention_guided_moesteer`
    - `comparison_notes`: free-form human annotation.
    - `preferred_condition`: which condition looked best by qualitative review.
    """

    concept: str
    evaluation_version: int
    evaluation_question: str
    full_prompt_text: str = ""
    prefix_conditioned_prompt_text: str = ""
    responses: Dict[str, str] = field(default_factory=dict)
    comparison_notes: str = ""
    preferred_condition: str = ""


@dataclass
class ManualReviewPlan:
    """A reproducible manual review bundle for one concept family.

    Meaning:
    - `condition_order` declares which named methods appear in every review case.
    - `condition_labels` maps those stable keys into human-readable report text.
    - `cases` stores the full concept-question cross product plus response slots.

    This keeps the report layer flexible: a run can compare baseline vs
    SteerMoE today, and later compare baseline vs SteerMoE vs attention steering
    without changing the plan schema again.
    """

    concept_type: str
    evaluation_family: str
    seed: int
    sampled_concepts: List[str]
    sampled_evaluation_versions: List[int]
    evaluation_questions_by_version: Dict[int, str]
    condition_order: List[str]
    condition_labels: Dict[str, str]
    cases: List[ManualReviewCase]


def extract_evaluation_question(prompt_template: str) -> str:
    """Extract the model-facing question from an evaluation prompt template.

    The imported upstream prompt files are evaluator instructions. Each file
    contains one quoted question or request that should be asked to the model
    whose outputs we want to inspect. This helper pulls out that quoted string so
    we can build manual test sheets.
    """
    match = QUESTION_PATTERN.search(prompt_template)
    if match is None:
        raise ValueError("Could not find a quoted model-facing question in the evaluation prompt.")
    return match.group(1)


def _resolve_condition_labels(
    condition_order: List[str],
    condition_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Fill in human-readable labels for all declared report conditions."""
    resolved = {key: DEFAULT_CONDITION_LABELS.get(key, key.replace("_", " ").title()) for key in condition_order}
    if condition_labels:
        resolved.update(condition_labels)
    return resolved


def build_manual_review_plan(
    concept_suite: ReferenceConceptSuite,
    concept_sample_size: int = 5,
    question_sample_size: int = 5,
    seed: int = 7,
    condition_order: Optional[List[str]] = None,
    condition_labels: Optional[Dict[str, str]] = None,
) -> ManualReviewPlan:
    """Build a reproducible manual inspection grid for named output conditions.

    Inputs:
    - `concept_suite`: one concept family plus its evaluation templates.
    - `concept_sample_size`: number of concepts to sample from the family.
    - `question_sample_size`: number of evaluation questions to sample.
    - `seed`: random seed used for reproducible sampling.
    - `condition_order`: ordered list of comparison conditions. Defaults to
      `["baseline", "steermoe"]` for the current faithful SteerMoE stage.
    - `condition_labels`: optional human-readable labels keyed by condition.

    Returns:
    - `ManualReviewPlan`: sampled concepts, sampled question versions, condition
      metadata, and the full concept-question cross product of review cases.
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

    condition_order = list(condition_order or DEFAULT_CONDITION_ORDER)
    if not condition_order:
        raise ValueError("condition_order must contain at least one condition.")
    condition_labels = _resolve_condition_labels(condition_order, condition_labels)

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
            full_prompt_text=evaluation_questions_by_version[version],
            prefix_conditioned_prompt_text=build_concept_conditioned_evaluation_prompt(
                concept_type=concept_suite.concept_type,
                concept_value=concept,
                evaluation_question=evaluation_questions_by_version[version],
            ),
            responses={condition: "" for condition in condition_order},
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
        condition_order=condition_order,
        condition_labels=condition_labels,
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
        "condition_order": list(plan.condition_order),
        "condition_labels": dict(plan.condition_labels),
        "cases": [
            {
                "concept": case.concept,
                "evaluation_version": case.evaluation_version,
                "evaluation_question": case.evaluation_question,
                "full_prompt_text": case.full_prompt_text,
                "prefix_conditioned_prompt_text": case.prefix_conditioned_prompt_text,
                "responses": dict(case.responses),
                "comparison_notes": case.comparison_notes,
                "preferred_condition": case.preferred_condition,
            }
            for case in plan.cases
        ],
    }


def _infer_condition_order_from_legacy_cases(cases: List[Dict[str, Any]]) -> List[str]:
    """Recover report conditions from older JSON plans with flat response fields."""
    condition_order = list(DEFAULT_CONDITION_ORDER)
    if any(case.get("attention_guided_moesteer_response", "").strip() for case in cases):
        condition_order.append("attention_guided_moesteer")
    return condition_order


def _responses_from_case_dict(case: Dict[str, Any], condition_order: List[str]) -> Dict[str, str]:
    """Normalize either new-style or legacy case payloads into `responses`."""
    if "responses" in case:
        responses = {str(key): str(value) for key, value in case["responses"].items()}
    else:
        responses = {
            "baseline": str(case.get("baseline_response", "")),
            "steermoe": str(case.get("moesteer_response", "")),
        }
        if "attention_guided_moesteer" in condition_order:
            responses["attention_guided_moesteer"] = str(
                case.get("attention_guided_moesteer_response", "")
            )

    for condition in condition_order:
        responses.setdefault(condition, "")
    return responses


def _prefix_conditioned_prompt_from_case(data: Dict[str, Any], case: Dict[str, Any]) -> str:
    """Recover the prefix-conditioned prompt for diagnostics and legacy plans."""
    expected_prefix_prompt = build_concept_conditioned_evaluation_prompt(
        concept_type=data["concept_type"],
        concept_value=case["concept"],
        evaluation_question=case["evaluation_question"],
    )
    stored_prefix_prompt = str(case.get("prefix_conditioned_prompt_text") or "")
    if stored_prefix_prompt:
        return stored_prefix_prompt

    legacy_full_prompt = str(case.get("full_prompt_text") or "")
    if legacy_full_prompt and legacy_full_prompt != case["evaluation_question"]:
        return legacy_full_prompt

    return expected_prefix_prompt


def manual_review_plan_from_dict(data: Dict[str, Any]) -> ManualReviewPlan:
    """Rehydrate a `ManualReviewPlan` from JSON-friendly serialized data.

    Backward compatibility:
    - older plans stored three flat response fields
    - newer plans store named responses in a single mapping
    """
    raw_cases = list(data["cases"])
    condition_order = list(data.get("condition_order") or _infer_condition_order_from_legacy_cases(raw_cases))
    condition_labels = _resolve_condition_labels(condition_order, data.get("condition_labels"))

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
        condition_order=condition_order,
        condition_labels=condition_labels,
        cases=[
            ManualReviewCase(
                concept=case["concept"],
                evaluation_version=int(case["evaluation_version"]),
                evaluation_question=case["evaluation_question"],
                full_prompt_text=str(
                    case.get("test_prompt_text")
                    or case.get("question_only_prompt_text")
                    or case.get("generation_prompt_text")
                    or case["evaluation_question"]
                ),
                prefix_conditioned_prompt_text=_prefix_conditioned_prompt_from_case(
                    data,
                    case,
                ),
                responses=_responses_from_case_dict(case, condition_order),
                comparison_notes=case.get("comparison_notes", ""),
                preferred_condition=case.get("preferred_condition", ""),
            )
            for case in raw_cases
        ],
    )


def load_manual_review_plan(path: Union[str, Path]) -> ManualReviewPlan:
    """Load a serialized manual review plan from disk."""
    return manual_review_plan_from_dict(json.loads(Path(path).read_text()))


def build_manual_review_markdown(plan: ManualReviewPlan) -> str:
    """Render a collaborator-friendly Markdown worksheet or filled report."""

    def display_response(value: str) -> str:
        return value if value.strip() else "Pending generation"

    lines = [
        "# Manual Review",
        "",
        f"- Concept family: **{plan.concept_type}**",
        f"- Evaluation family: **{plan.evaluation_family}**",
        f"- Sampling seed: **{plan.seed}**",
        f"- Number of sampled concepts: **{len(plan.sampled_concepts)}**",
        f"- Number of sampled questions: **{len(plan.sampled_evaluation_versions)}**",
        f"- Conditions per case: **{', '.join(plan.condition_labels[condition] for condition in plan.condition_order)}**",
        f"- Number of comparison cases: **{len(plan.cases)}**",
        "",
        "## How To Read This Report",
        "",
        "- Same-model baseline and `+ SteerMoE` columns use the same MoE model when both are present.",
        "- Any reference-model column is unsteered question-only context; it is not the steering control.",
        "- `Question-only test prompt` is the exact prompt sent to each generation condition.",
        "- The concept prefix is intentionally omitted during this steering test.",
        "- `Prefix-conditioned diagnostic prompt` is shown for context only.",
        "- Baseline/reference columns = question-only prompt, no steering.",
        "- `+ SteerMoE` columns = question-only prompt plus router bias from the saved SteerMoE plan.",
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
            lines.extend(
                [
                    f"### Eval v{version}",
                    "",
                    f"- Question: {matching_case.evaluation_question}",
                    f"- Question-only test prompt: {matching_case.full_prompt_text}",
                    f"- Prefix-conditioned diagnostic prompt (not used here): {matching_case.prefix_conditioned_prompt_text}",
                ]
            )
            for condition in plan.condition_order:
                lines.extend(
                    [
                        f"- {plan.condition_labels[condition]} response:",
                        f"  {display_response(matching_case.responses.get(condition, ''))}",
                    ]
                )
            lines.extend(
                [
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
    companion_report: str = "",
) -> str:
    """Render a browsable HTML report for qualitative response inspection."""

    def display_response(value: str) -> str:
        return html.escape(value) if value.strip() else "<span class='pending'>Pending generation</span>"

    legend_items = "".join(
        f"<li><strong>{html.escape(plan.condition_labels[condition])}</strong> &mdash; generated output for condition key <code>{html.escape(condition)}</code>.</li>"
        for condition in plan.condition_order
    )

    condition_colors = [
        ("#555", "#f6f8fa"),
        ("#0550ae", "#eef5ff"),
        ("#cf222e", "#fff1f0"),
        ("#8250df", "#f5f0ff"),
    ]

    concept_sections: List[str] = []
    for concept in plan.sampled_concepts:
        concept_cases = [case for case in plan.cases if case.concept == concept]
        for rank, case in enumerate(concept_cases, start=1):
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

            condition_blocks = []
            for index, condition in enumerate(plan.condition_order):
                color, background = condition_colors[index % len(condition_colors)]
                filled = bool(case.responses.get(condition, "").strip())
                badge = (
                    "<span class='badge b1'>Filled</span>"
                    if filled
                    else "<span class='badge b0'>Pending</span>"
                )
                condition_blocks.append(
                    f"""
                    <div class='condition' style='border-left:4px solid {color};background:{background};'>
                      <p class='lbl' style='color:{color};'>{html.escape(plan.condition_labels[condition])} {badge}</p>
                      <pre>{display_response(case.responses.get(condition, ''))}</pre>
                    </div>
                    """
                )

            concept_sections.append(
                f"""
                <section>
                <h2>#{rank} &mdash; <strong>{html.escape(concept)}</strong> &middot; prompt v{case.evaluation_version}</h2>
                <p><strong>Question:</strong> {html.escape(case.evaluation_question)}</p>
                <p><strong>Question-only test prompt:</strong> <code>{html.escape(case.full_prompt_text)}</code></p>
                <p><strong>Prefix-conditioned diagnostic prompt, not used here:</strong> <code>{html.escape(case.prefix_conditioned_prompt_text)}</code></p>
                <div class='conditions'>
                  {''.join(condition_blocks)}
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
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:2rem auto;line-height:1.5;color:#24292f;padding:0 1rem;}}
h1{{font-size:1.45rem;}} h2{{font-size:1.05rem;margin:2rem 0 .5rem;}}
section{{border:1px solid #d0d7de;border-radius:8px;padding:1rem;margin-bottom:1.5rem;background:#fafafa;}}
.conditions{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;}}
.condition{{min-width:0;padding:.75rem;border-radius:8px;}}
.lbl{{font-weight:700;font-size:.9rem;margin-bottom:.3rem;}}
pre{{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;padding:.6rem;border-radius:6px;font-size:.85rem;margin:.3rem 0 .1rem;min-height:4.5rem;}}
.badge{{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-weight:600;font-size:.82rem;margin-left:.3rem;}}
.b1{{background:#d4edda;color:#155724;}} .b0{{background:#f8d7da;color:#721c24;}}
.meta{{color:#57606a;font-size:.85rem;}}
table{{border-collapse:collapse;margin:1rem 0;}}
td,th{{padding:.4rem .8rem;border:1px solid #d0d7de;text-align:left;}}
th{{background:#f6f8fa;}}
.note{{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:.6rem 1rem;margin:1rem 0;font-size:.9rem;}}
.pending{{color:#8a5a00;font-style:italic;}}
code{{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;}}
  </style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class='note'>
<strong>What are we comparing?</strong>
<ul style='margin:.4rem 0 .2rem;padding-left:1.3rem;'>
{legend_items}
</ul>
<strong>Important:</strong> same-model baseline and <em>+ SteerMoE</em> columns use the same MoE model when both are present; reference-model columns are unsteered context, not the steering control.<br>
<strong>Question-only test prompt</strong> = the exact text each generation condition receives, for example:<br>
<code>What is the scariest thing in the world? Answer in 20 words or less.</code><br>
<strong>Prefix-conditioned diagnostic prompt</strong> shows the omitted concept prefix for context, but is not sent in this steering test.<br>
<strong>Baseline/reference columns</strong> = question-only prompt, no steering.<br>
<strong>+ SteerMoE columns</strong> = question-only prompt plus router bias from the saved SteerMoE plan.<br>
<strong>What is being evaluated:</strong> whether SteerMoE makes the response more concept-faithful without making it worse or less coherent.
</div>
<p class='meta'>Concept family: <code>{html.escape(plan.concept_type)}</code> &middot; Evaluation family: <code>{html.escape(plan.evaluation_family)}</code> &middot; Seed: <code>{plan.seed}</code> &middot; Companion report: <code>{html.escape(companion_report) if companion_report else 'not linked'}</code></p>
{''.join(concept_sections)}
</body>
</html>
"""
