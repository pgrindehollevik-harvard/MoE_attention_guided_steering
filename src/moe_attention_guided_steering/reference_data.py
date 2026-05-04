from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


EVALUATION_FAMILY_BY_CONCEPT_TYPE = {
    "fears": "phobia",
    "moods": "mood",
    "personas": "persona",
    "personalities": "personality",
    "places": "topophile",
}


@dataclass
class ReferenceDataBundle:
    """All imported text assets from the attention-guided steering repo.

    These files are useful for experiment setup, evaluation prompts, and concept
    catalogs. The steering runners collect model-specific router traces from
    these text inputs at experiment time.
    """

    concept_values_by_type: Dict[str, List[str]]
    evaluation_prompts_by_family: Dict[str, Dict[int, str]]
    general_statements_by_class: Dict[str, List[str]]


@dataclass
class ReferenceConceptSuite:
    """The subset of imported data needed for one concept family.

    This is the adapter layer between the vendored text files and the current
    OLMoE runner plus the future attention-collector entrypoint.
    """

    concept_type: str
    evaluation_family: str
    concepts: List[str]
    evaluation_prompts_by_version: Dict[int, str]
    general_statements_by_class: Dict[str, List[str]]


def _read_nonempty_lines(path: Path) -> List[str]:
    """Read a text file and discard blank lines.

    The upstream data is line oriented. Keeping the parsing rule tiny and explicit
    makes it easy to audit exactly what becomes part of the in-memory catalog.
    """
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _parse_evaluation_prompt_path(path: Path) -> Tuple[str, int]:
    """Extract the evaluation family name and version from one filename.

    Example: `persona_eval_v3.txt` becomes `("persona", 3)`.
    """
    family, version_text = path.stem.rsplit("_eval_v", 1)
    return family, int(version_text)


def load_reference_data(data_dir: str = "data") -> ReferenceDataBundle:
    """Load the imported attention-guided steering text assets.

    The directory structure mirrors the upstream repo:

    - `concepts/`: concept names grouped by family,
    - `general_statements/`: shared statement pools,
    - `evaluation_prompts/`: evaluation instructions keyed by family and version.
    """
    root = Path(data_dir)
    concepts_dir = root / "concepts"
    evaluation_dir = root / "evaluation_prompts"
    statements_dir = root / "general_statements"

    concept_values_by_type = {
        path.stem: _read_nonempty_lines(path)
        for path in sorted(concepts_dir.glob("*.txt"))
    }

    evaluation_prompts_by_family: Dict[str, Dict[int, str]] = {}
    for path in sorted(evaluation_dir.glob("*.txt")):
        family, version = _parse_evaluation_prompt_path(path)
        evaluation_prompts_by_family.setdefault(family, {})[version] = path.read_text().strip()

    general_statements_by_class = {
        path.stem: _read_nonempty_lines(path)
        for path in sorted(statements_dir.glob("*.txt"))
    }

    return ReferenceDataBundle(
        concept_values_by_type=concept_values_by_type,
        evaluation_prompts_by_family=evaluation_prompts_by_family,
        general_statements_by_class=general_statements_by_class,
    )


def summarize_reference_data(bundle: ReferenceDataBundle) -> Dict[str, int]:
    """Return a compact summary for CLI inspection and docs."""
    return {
        "num_concept_families": len(bundle.concept_values_by_type),
        "num_total_concepts": sum(
            len(concepts) for concepts in bundle.concept_values_by_type.values()
        ),
        "num_evaluation_prompt_families": len(bundle.evaluation_prompts_by_family),
        "num_evaluation_prompt_versions": sum(
            len(versions) for versions in bundle.evaluation_prompts_by_family.values()
        ),
        "num_general_statement_classes": len(bundle.general_statements_by_class),
        "num_total_general_statements": sum(
            len(statements) for statements in bundle.general_statements_by_class.values()
        ),
    }


def load_reference_concept_suite(
    concept_type: str,
    data_dir: str = "data",
) -> ReferenceConceptSuite:
    """Load one concept family together with its matching evaluation prompts.

    The upstream repo uses slightly different names for concept files and
    evaluation files, such as `fears` paired with `phobia_eval_v*.txt`. This
    helper captures that mapping in one place so later collector code does not
    need to remember it ad hoc.
    """
    bundle = load_reference_data(data_dir)

    if concept_type not in bundle.concept_values_by_type:
        available = ", ".join(sorted(bundle.concept_values_by_type))
        raise ValueError(f"Unknown concept type '{concept_type}'. Available types: {available}.")

    evaluation_family = EVALUATION_FAMILY_BY_CONCEPT_TYPE.get(concept_type)
    if evaluation_family is None:
        raise ValueError(
            f"Concept type '{concept_type}' does not yet have a mapped evaluation prompt family."
        )

    if evaluation_family not in bundle.evaluation_prompts_by_family:
        raise ValueError(
            f"No evaluation prompts were found for mapped family '{evaluation_family}'."
        )

    return ReferenceConceptSuite(
        concept_type=concept_type,
        evaluation_family=evaluation_family,
        concepts=list(bundle.concept_values_by_type[concept_type]),
        evaluation_prompts_by_version=dict(bundle.evaluation_prompts_by_family[evaluation_family]),
        general_statements_by_class={
            label: list(statements)
            for label, statements in bundle.general_statements_by_class.items()
        },
    )
