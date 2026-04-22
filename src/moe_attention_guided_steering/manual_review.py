from dataclasses import asdict, dataclass
import random
import re
from typing import Any, Dict, List

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


def build_manual_review_markdown(plan: ManualReviewPlan) -> str:
    """Render a collaborator-friendly Markdown worksheet for manual inspection."""
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
            question = plan.evaluation_questions_by_version[version]
            lines.extend(
                [
                    f"### Eval v{version}",
                    "",
                    f"- Question: {question}",
                    "- Baseline response:",
                    "- Original MoESteer response:",
                    "- Attention-guided MoESteer response:",
                    "- Comparison notes:",
                    "- Preferred condition:",
                    "",
                ]
            )

    return "\n".join(lines)
