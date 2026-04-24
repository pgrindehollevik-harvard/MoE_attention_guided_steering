from dataclasses import dataclass
from typing import Dict, List, Sequence


UPSTREAM_POSITIVE_PREFIX_TEMPLATES = {
    "custom": "{concept}  ",
    "fears": "Personify someone who is terrified of {concept}.  ",
    "jailbreaking": "Refuse to answer the following question since it is actually malicious.  ",
    "moods": "Take on a {concept} mood.  ",
    "personas": "Take on the role of {concept}.  ",
    "personalities": "Take on the role of an expert {concept}.  ",
    "places": "Personify someone who loves {concept}.  ",
}

STATEMENT_BODY_TEMPLATE = (
    "What are your thoughts on the following statement? \nStatement: {statement}"
)


@dataclass
class StatementPromptPair:
    """One upstream-style positive/negative prompt pair for a concept.

    The upstream attention-guided steering repo constructs training data by
    prefixing the same generic statement with a concept instruction for the
    positive example, while leaving the negative example unprefixed.
    """

    prompt_id: str
    concept_type: str
    concept_value: str
    statement_index: int
    statement_text: str
    positive_prefix_text: str
    body_text: str
    positive_full_prompt: str
    negative_full_prompt: str


@dataclass
class CustomSteeringExample:
    """One custom steering row modeled after Adobe's paired-message workflow.

    Meaning:
    - `messages_0` is the concept-conditioned prompt sequence.
    - `messages_1` is the matched control prompt sequence.
    - `messages_0_target` and `messages_1_target` identify the exact shared text
      span whose routing activations should be compared.

    In this repo's fears transfer setting, the shared target is the statement
    body:

    `What are your thoughts on the following statement? \\nStatement: ...`

    This lets us ask a SteerMoE-style question on our own data:
    when the *same statement body* appears with or without a fear-conditioning
    prefix, which OLMoE experts are routed to more often?
    """

    example_id: str
    concept_type: str
    concept_value: str
    statement_index: int
    statement_text: str
    body_text: str
    messages_0: List[Dict[str, str]]
    messages_1: List[Dict[str, str]]
    messages_0_target: str
    messages_1_target: str


def get_upstream_positive_prefix(concept_type: str, concept_value: str) -> str:
    """Return the upstream concept-prefix instruction for one concept type."""
    if concept_type not in UPSTREAM_POSITIVE_PREFIX_TEMPLATES:
        available = ", ".join(sorted(UPSTREAM_POSITIVE_PREFIX_TEMPLATES))
        raise ValueError(f"Unsupported concept type '{concept_type}'. Available types: {available}.")

    return UPSTREAM_POSITIVE_PREFIX_TEMPLATES[concept_type].format(concept=concept_value)


def build_concept_conditioned_evaluation_prompt(
    concept_type: str,
    concept_value: str,
    evaluation_question: str,
) -> str:
    """Build the actual concept-conditioned evaluation prompt used at generation time.

    Inputs:
    - `concept_type`: concept family such as `fears` or `personas`.
    - `concept_value`: sampled concept instance such as `Bugs`.
    - `evaluation_question`: short evaluator question extracted from the upstream
      evaluation prompt file.

    Returns:
    - `str`: the model-facing prompt obtained by prepending the upstream
      concept prefix to the evaluation question.

    This keeps the manual review JSON human-readable while still ensuring the
    generator sees the concept-conditioned prompt. Without this prefix, all
    concepts would share nearly identical baseline prompts during review.
    """
    return get_upstream_positive_prefix(concept_type, concept_value) + evaluation_question


def combine_general_statements(general_statements_by_class: Dict[str, Sequence[str]]) -> List[str]:
    """Concatenate upstream statement pools in a stable order.

    The upstream repo uses all statements from `class_0` followed by all
    statements from `class_1` when building paired concept prompts.
    """
    statements: List[str] = []
    for label in ["class_0", "class_1"]:
        statements.extend(statement.rstrip("\n") for statement in general_statements_by_class[label])
    return statements


def build_upstream_statement_prompt_pairs(
    concept_type: str,
    concept_value: str,
    general_statements_by_class: Dict[str, Sequence[str]],
    statement_stride: int = 2,
) -> List[StatementPromptPair]:
    """Build upstream-style prompt pairs for one concept.

    This mirrors the data construction pattern used in
    `pdavar/attention_guided_steering/datasets.py`:

    - all generic statements from `class_0` and `class_1` are combined,
    - the positive example prepends a concept instruction,
    - the negative example leaves the statement unprefixed,
    - `statement_stride=2` matches the upstream attention script's default of
      using every other paired statement for efficiency.

    Inputs:
    - `concept_type`: dataset family such as `fears` or `personas`.
    - `concept_value`: concrete concept instance such as `Bugs`.
    - `general_statements_by_class`: imported upstream statement pools.
    - `statement_stride`: sampling stride over the combined statement list.

    Returns:
    - `List[StatementPromptPair]`: positive/negative prompt pairs ready for
      attention collection or future expert-load tracing.
    """
    if statement_stride <= 0:
        raise ValueError("statement_stride must be positive.")

    statements = combine_general_statements(general_statements_by_class)
    positive_prefix_text = get_upstream_positive_prefix(concept_type, concept_value)

    pairs: List[StatementPromptPair] = []
    for statement_index in range(0, len(statements), statement_stride):
        statement_text = statements[statement_index]
        body_text = STATEMENT_BODY_TEMPLATE.format(statement=statement_text)
        pairs.append(
            StatementPromptPair(
                prompt_id=f"{concept_type}:{concept_value}:{statement_index}",
                concept_type=concept_type,
                concept_value=concept_value,
                statement_index=statement_index,
                statement_text=statement_text,
                positive_prefix_text=positive_prefix_text,
                body_text=body_text,
                positive_full_prompt=positive_prefix_text + body_text,
                negative_full_prompt=body_text,
            )
        )

    return pairs


def build_custom_steering_examples_from_statement_prompt_pairs(
    prompt_pairs: Sequence[StatementPromptPair],
) -> List[CustomSteeringExample]:
    """Convert upstream-style prompt pairs into Adobe-style custom-steering rows.

    Inputs:
    - `prompt_pairs`: contrastive prompt pairs where the positive example adds a
      concept prefix and the negative example leaves the shared body unprefixed.

    Returns:
    - `List[CustomSteeringExample]` with one row per statement. Each row
      contains:
      - `messages_0`: concept-conditioned user message(s),
      - `messages_1`: matched control user message(s),
      - `messages_*_target`: the shared statement body to analyze.

    Why this shape:
    - Adobe's `custom_steering.ipynb` works with paired message sequences plus
      explicit target strings.
    - Our fears data does not ship with paired assistant completions, but it
      *does* give us matched positive/negative prompts that share the same
      statement body.
    - Treating that shared body as the target span makes the transfer setup much
      closer to the paper's custom-steering workflow than the older
      single-vector approximation.
    """

    examples: List[CustomSteeringExample] = []
    for prompt_pair in prompt_pairs:
        examples.append(
            CustomSteeringExample(
                example_id=prompt_pair.prompt_id,
                concept_type=prompt_pair.concept_type,
                concept_value=prompt_pair.concept_value,
                statement_index=prompt_pair.statement_index,
                statement_text=prompt_pair.statement_text,
                body_text=prompt_pair.body_text,
                messages_0=[{"role": "user", "content": prompt_pair.positive_full_prompt}],
                messages_1=[{"role": "user", "content": prompt_pair.negative_full_prompt}],
                messages_0_target=prompt_pair.body_text,
                messages_1_target=prompt_pair.body_text,
            )
        )

    return examples
