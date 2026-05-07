#!/usr/bin/env python3
"""Run a question-only Llama reference vs Mixtral baseline/SteerMoE review."""

from dataclasses import asdict
from pathlib import Path
import argparse
import gc
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import load_hf_model_resources  # noqa: E402
from moe_attention_guided_steering.generation import generate_unsteered_response  # noqa: E402
from moe_attention_guided_steering.io_utils import write_json  # noqa: E402
from moe_attention_guided_steering.manual_review import (  # noqa: E402
    ManualReviewPlan,
    build_manual_review_html,
    build_manual_review_markdown,
    load_manual_review_plan,
    manual_review_plan_to_dict,
)
from moe_attention_guided_steering.mixtral_backend import (  # noqa: E402
    build_steermoe_activation_table_from_paired_traces,
    build_steermoe_replication_plan,
    collect_mixtral_paired_routing_traces,
    fill_manual_review_plan_with_mixtral_generations,
)
from moe_attention_guided_steering.reference_data import load_reference_data  # noqa: E402
from moe_attention_guided_steering.upstream_prompt_datasets import (  # noqa: E402
    build_concept_conditioned_evaluation_prompt,
    build_custom_steering_examples_from_statement_prompt_pairs,
    build_upstream_statement_prompt_pairs,
)


MIXTRAL_BASELINE_CONDITION = "mixtral_baseline"
MIXTRAL_STEERMOE_CONDITION = "mixtral_steermoe"
LLAMA_REFERENCE_CONDITION = "reference_model"


def _clear_cuda_cache() -> None:
    """Release model memory before loading the optional reference model."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is installed on GPU runs
        pass


def _limit_plan_cases(plan: ManualReviewPlan, limit_cases: int) -> ManualReviewPlan:
    """Keep the first N cases and trim sampled concept/version metadata to match."""
    plan.cases = plan.cases[:limit_cases]
    seen_concepts = []
    seen_versions = []
    for case in plan.cases:
        if case.concept not in seen_concepts:
            seen_concepts.append(case.concept)
        if case.evaluation_version not in seen_versions:
            seen_versions.append(case.evaluation_version)
    plan.sampled_concepts = seen_concepts
    plan.sampled_evaluation_versions = seen_versions
    plan.evaluation_questions_by_version = {
        version: plan.evaluation_questions_by_version[version]
        for version in seen_versions
    }
    return plan


def _prepare_review_plan(
    plan: ManualReviewPlan,
    include_llama_reference: bool = True,
    prompt_mode: str = "question_only",
) -> ManualReviewPlan:
    """Prepare report conditions and choose visible prompt text."""
    if prompt_mode not in {"question_only", "prefix_conditioned"}:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    plan.condition_order = []
    if include_llama_reference:
        plan.condition_order.append(LLAMA_REFERENCE_CONDITION)
    plan.condition_order.extend(
        [
            MIXTRAL_BASELINE_CONDITION,
            MIXTRAL_STEERMOE_CONDITION,
        ]
    )
    prompt_label = "question only" if prompt_mode == "question_only" else "prefix conditioned"
    plan.condition_labels = {
        LLAMA_REFERENCE_CONDITION: f"Llama 3.1 8B baseline ({prompt_label})",
        MIXTRAL_BASELINE_CONDITION: f"Mixtral 8x7B baseline ({prompt_label})",
        MIXTRAL_STEERMOE_CONDITION: f"Mixtral 8x7B + SteerMoE ({prompt_label})",
    }
    for case in plan.cases:
        if not case.prefix_conditioned_prompt_text:
            case.prefix_conditioned_prompt_text = build_concept_conditioned_evaluation_prompt(
                concept_type=plan.concept_type,
                concept_value=case.concept,
                evaluation_question=case.evaluation_question,
            )
        if prompt_mode == "question_only":
            case.full_prompt_text = case.evaluation_question
        else:
            case.full_prompt_text = case.prefix_conditioned_prompt_text
        existing = dict(case.responses)
        case.responses = {condition: existing.get(condition, "") for condition in plan.condition_order}
    return plan


def _fill_llama_reference_generations(
    plan: ManualReviewPlan,
    resources,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> ManualReviewPlan:
    """Fill the unsteered Llama reference column from question-only prompts."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    for case in tqdm(plan.cases, desc="Generating Llama reference responses"):
        case.responses[LLAMA_REFERENCE_CONDITION] = generate_unsteered_response(
            prompt_text=case.full_prompt_text,
            resources=resources,
            prompt_format="chat",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    return plan


def main() -> None:
    """Run the Mixtral question-only steering experiment.

    The final report columns are:
    1. Llama 3.1 8B baseline, question-only and unsteered.
    2. Mixtral 8x7B baseline, question-only and unsteered.
    3. Mixtral 8x7B + SteerMoE, question-only with router-logit bias.
    """
    parser = argparse.ArgumentParser(
        description="Run question-only Mixtral baseline vs Mixtral SteerMoE."
    )
    parser.add_argument(
        "--mixtral-model-id",
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Hugging Face model id for the Mixtral checkpoint.",
    )
    parser.add_argument(
        "--mixtral-model-tag",
        default="mixtral_8x7b_instruct_v0_1",
        help="Filesystem-friendly Mixtral model tag.",
    )
    parser.add_argument(
        "--include-llama-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a Llama 3.1 8B unsteered question-only reference column. Enabled by default.",
    )
    parser.add_argument(
        "--llama-reference-model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id for the optional Llama reference column.",
    )
    parser.add_argument(
        "--llama-reference-model-tag",
        default="llama_3_1_8b_instruct",
        help="Filesystem-friendly model tag for the optional Llama reference column.",
    )
    parser.add_argument(
        "--llama-reference-attn-implementation",
        default="eager",
        help="Transformers attention implementation for the optional Llama reference model.",
    )
    parser.add_argument(
        "--load-llama-reference-in-8bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the optional Llama reference model in 8-bit. Disabled by default.",
    )
    parser.add_argument(
        "--plan-json",
        default="outputs/manual_fear_review/manual_review_plan.json",
        help="Manual review plan JSON describing which concepts/questions to run.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the imported upstream concept and statement files.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/mixtral_steermoe_fears_seed7_question_only_with_llama",
        help="Experiment directory where datasets, steering plans, and reports should be written.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of concept/question cases.",
    )
    parser.add_argument(
        "--statement-stride",
        type=int,
        default=2,
        help="Stride over the upstream statement pool when building concept prompt pairs.",
    )
    parser.add_argument(
        "--readout-target",
        default="statement_body",
        choices=["statement_body"],
        help="Which explicit target string inside each paired prompt should define the routing readout span.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="question_only",
        choices=["question_only", "prefix_conditioned"],
        help="Use bare evaluation questions or visible concept-conditioned prompts during generation.",
    )
    parser.add_argument(
        "--top-positive-experts",
        type=int,
        default=8,
        help="Number of globally strongest positive-risk Mixtral experts to activate.",
    )
    parser.add_argument(
        "--top-negative-experts",
        type=int,
        default=8,
        help="Number of globally strongest negative-risk Mixtral experts to deactivate.",
    )
    parser.add_argument(
        "--minimum-abs-risk-difference",
        type=float,
        default=0.01,
        help="Minimum absolute risk difference required for an expert to be eligible for steering.",
    )
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help=(
            "Additive router-logit bias magnitude, or epsilon for "
            "--steering-rule=paper."
        ),
    )
    parser.add_argument(
        "--steering-rule",
        default="additive_bias",
        choices=["additive_bias", "paper"],
        help=(
            "Router intervention rule. `paper` applies log-softmax, then sets "
            "activated experts to s_max + epsilon and deactivated experts to "
            "s_min - epsilon for every router row."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum number of new tokens to generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. 0.0 means greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus-sampling cutoff used only when temperature > 0.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map argument, such as auto or cuda:0.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype name passed to from_pretrained, such as bfloat16 or float16.",
    )
    parser.add_argument(
        "--mixtral-attn-implementation",
        default="sdpa",
        help="Transformers attention implementation for Mixtral.",
    )
    parser.add_argument(
        "--load-mixtral-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load Mixtral with bitsandbytes 4-bit quantization. Disabled by default on ORCD.",
    )
    parser.add_argument(
        "--load-mixtral-in-8bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load Mixtral with bitsandbytes 8-bit quantization. Enabled by default on ORCD.",
    )
    parser.add_argument(
        "--bnb-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bitsandbytes to place overflow Mixtral modules on CPU. Enabled by default.",
    )
    args = parser.parse_args()
    if args.load_mixtral_in_4bit and args.load_mixtral_in_8bit:
        parser.error("Choose either --load-mixtral-in-4bit or --load-mixtral-in-8bit, not both.")

    plan = load_manual_review_plan(args.plan_json)
    if args.limit_cases is not None:
        plan = _limit_plan_cases(plan, args.limit_cases)
    plan = _prepare_review_plan(
        plan,
        include_llama_reference=args.include_llama_reference,
        prompt_mode=args.prompt_mode,
    )

    reference_data = load_reference_data(args.data_dir)
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "custom_steering_datasets"
    trace_dir = output_dir / "routing_traces"
    activation_dir = output_dir / "activation_tables"
    steering_dir = output_dir / "steering_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    steering_dir.mkdir(parents=True, exist_ok=True)

    mixtral_resources = load_hf_model_resources(
        model_id=args.mixtral_model_id,
        model_tag=args.mixtral_model_tag,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_mixtral_in_4bit,
        load_in_8bit=args.load_mixtral_in_8bit,
        bnb_cpu_offload=args.bnb_cpu_offload,
        attn_implementation=args.mixtral_attn_implementation,
        infer_attention_suffix_tokens=False,
    )

    steermoe_plans_by_concept = {}
    for concept in plan.sampled_concepts:
        print(f"=== Building Mixtral SteerMoE plan for {plan.concept_type}:{concept} ===")
        statement_prompt_pairs = build_upstream_statement_prompt_pairs(
            concept_type=plan.concept_type,
            concept_value=concept,
            general_statements_by_class=reference_data.general_statements_by_class,
            statement_stride=args.statement_stride,
        )
        custom_examples = build_custom_steering_examples_from_statement_prompt_pairs(
            statement_prompt_pairs
        )
        paired_traces = collect_mixtral_paired_routing_traces(
            examples=custom_examples,
            resources=mixtral_resources,
            target_name=args.readout_target,
        )
        activation_table = build_steermoe_activation_table_from_paired_traces(
            paired_traces=paired_traces
        )
        steering_plan = build_steermoe_replication_plan(
            concept=concept,
            concept_type=plan.concept_type,
            model_id=mixtral_resources.model_id,
            model_tag=mixtral_resources.model_tag,
            target_name=args.readout_target,
            paired_traces=paired_traces,
            activation_table=activation_table,
            top_positive_experts=args.top_positive_experts,
            top_negative_experts=args.top_negative_experts,
            minimum_abs_risk_difference=args.minimum_abs_risk_difference,
        )
        steermoe_plans_by_concept[concept] = steering_plan

        write_json(
            [asdict(example) for example in custom_examples],
            dataset_dir / f"{mixtral_resources.model_tag}_{concept}_custom_steering_dataset.json",
        )
        write_json(
            [asdict(trace) for trace in paired_traces],
            trace_dir / f"{mixtral_resources.model_tag}_{concept}_routing_traces.json",
        )
        write_json(
            [asdict(score) for score in activation_table],
            activation_dir / f"{mixtral_resources.model_tag}_{concept}_activation_table.json",
        )
        write_json(
            asdict(steering_plan),
            steering_dir / f"{mixtral_resources.model_tag}_{concept}_steering_plan.json",
        )

    plan = fill_manual_review_plan_with_mixtral_generations(
        plan=plan,
        resources=mixtral_resources,
        steermoe_plans_by_concept=steermoe_plans_by_concept,
        steering_coefficient=args.steering_coefficient,
        steering_rule=args.steering_rule,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        baseline_condition=MIXTRAL_BASELINE_CONDITION,
        steermoe_condition=MIXTRAL_STEERMOE_CONDITION,
    )

    del mixtral_resources
    _clear_cuda_cache()

    if args.include_llama_reference:
        llama_reference_resources = load_hf_model_resources(
            model_id=args.llama_reference_model_id,
            model_tag=args.llama_reference_model_tag,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            load_in_8bit=args.load_llama_reference_in_8bit,
            bnb_cpu_offload=args.bnb_cpu_offload,
            attn_implementation=args.llama_reference_attn_implementation,
            infer_attention_suffix_tokens=False,
        )
        try:
            plan = _fill_llama_reference_generations(
                plan=plan,
                resources=llama_reference_resources,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        finally:
            del llama_reference_resources
            _clear_cuda_cache()

    write_json(manual_review_plan_to_dict(plan), output_dir / "manual_review_plan.json")
    write_json(
        {
            "prompt_mode": args.prompt_mode,
            "prompt_contract": (
                "Mixtral baseline and Mixtral + SteerMoE both receive only "
                "the evaluation question; the concept prefix is omitted at "
                "test time"
            ),
            "training_signal": (
                "Mixtral SteerMoE plans are learned from prefix-conditioned vs "
                "unprefixed routing traces over matched statement bodies"
            ),
            "mixtral_model_id": args.mixtral_model_id,
            "mixtral_model_tag": args.mixtral_model_tag,
            "mixtral_loaded_in_4bit": args.load_mixtral_in_4bit,
            "mixtral_loaded_in_8bit": args.load_mixtral_in_8bit,
            "bnb_cpu_offload": args.bnb_cpu_offload,
            "include_llama_reference": args.include_llama_reference,
            "llama_reference_model_id": (
                args.llama_reference_model_id if args.include_llama_reference else ""
            ),
            "llama_reference_model_tag": (
                args.llama_reference_model_tag if args.include_llama_reference else ""
            ),
            "llama_reference_loaded_in_8bit": (
                args.load_llama_reference_in_8bit if args.include_llama_reference else False
            ),
            "steering_coefficient": args.steering_coefficient,
            "steering_rule": args.steering_rule,
            "top_positive_experts": args.top_positive_experts,
            "top_negative_experts": args.top_negative_experts,
            "minimum_abs_risk_difference": args.minimum_abs_risk_difference,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        output_dir / "generation_metadata.json",
    )
    (output_dir / "qualitative_review.md").write_text(build_manual_review_markdown(plan))
    (output_dir / "qualitative_review.html").write_text(
        build_manual_review_html(
            plan,
            title="Question-Only Mixtral SteerMoE Review With Llama Reference",
        )
    )

    print(f"Wrote experiment bundle to {output_dir}")
    print(f"- {output_dir / 'manual_review_plan.json'}")
    print(f"- {output_dir / 'qualitative_review.md'}")
    print(f"- {output_dir / 'qualitative_review.html'}")


if __name__ == "__main__":
    main()
