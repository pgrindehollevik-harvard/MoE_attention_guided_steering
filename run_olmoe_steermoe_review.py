#!/usr/bin/env python3
"""Run a question-only baseline-vs-SteerMoE qualitative review on OLMoE."""

from dataclasses import asdict
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.io_utils import write_json  # noqa: E402
from moe_attention_guided_steering.manual_review import (  # noqa: E402
    build_manual_review_html,
    build_manual_review_markdown,
    load_manual_review_plan,
    manual_review_plan_to_dict,
)
from moe_attention_guided_steering.olmoe_backend import (  # noqa: E402
    build_steermoe_activation_table_from_paired_traces,
    build_steermoe_replication_plan,
    collect_olmoe_paired_routing_traces,
    fill_manual_review_plan_with_steermoe_generations,
)
from moe_attention_guided_steering.reference_data import load_reference_data  # noqa: E402
from moe_attention_guided_steering.attention_collection import load_hf_model_resources  # noqa: E402
from moe_attention_guided_steering.upstream_prompt_datasets import (  # noqa: E402
    build_concept_conditioned_evaluation_prompt,
    build_custom_steering_examples_from_statement_prompt_pairs,
    build_upstream_statement_prompt_pairs,
)


def main() -> None:
    """Run the stage-1 OLMoE transfer experiment: baseline vs SteerMoE.

    High-level stages:
    1. Load the sampled manual review cases.
    2. For each concept, build upstream-style positive/negative statement pairs.
    3. Convert those pairs into Adobe-style custom steering examples.
    4. Collect OLMoE router traces over the shared statement-body target.
    5. Build the SteerMoE risk-difference table and select global experts.
    6. Generate baseline and SteerMoE answers from question-only prompts.
    7. Write the filled JSON plan plus browsable HTML/Markdown reports.
    """
    parser = argparse.ArgumentParser(
        description="Run question-only baseline vs SteerMoE on OLMoE using a custom-steering pipeline adapted to our fears data."
    )
    parser.add_argument(
        "--model-id",
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="Hugging Face model id for the OLMoE checkpoint.",
    )
    parser.add_argument(
        "--model-tag",
        default="olmoe_1b_7b_0125_instruct",
        help="Filesystem-friendly model tag used in output filenames.",
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
        default="experiments/olmoe_steermoe_fears_seed7_question_only",
        help="Experiment directory where datasets, steering plans, and reports should be written.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of concept/question cases to run.",
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
        "--top-positive-experts",
        type=int,
        default=8,
        help="Number of globally strongest positive-risk experts to activate.",
    )
    parser.add_argument(
        "--top-negative-experts",
        type=int,
        default=8,
        help="Number of globally strongest negative-risk experts to deactivate.",
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
        help="Maximum absolute router-logit bias magnitude used during generation.",
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
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation. Kept explicit for model loading consistency.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load OLMoE with optional bitsandbytes 4-bit quantization when available.",
    )
    args = parser.parse_args()

    plan = load_manual_review_plan(args.plan_json)
    if plan.condition_order != ["baseline", "steermoe"]:
        plan.condition_order = ["baseline", "steermoe"]
        plan.condition_labels = {
            "baseline": "OLMoE baseline (question only)",
            "steermoe": "OLMoE + SteerMoE (question only)",
        }
        for case in plan.cases:
            baseline_value = case.responses.get("baseline", "")
            steermoe_value = case.responses.get("steermoe", "")
            case.responses = {
                "baseline": baseline_value,
                "steermoe": steermoe_value,
            }
    else:
        plan.condition_labels.update(
            {
                "baseline": "OLMoE baseline (question only)",
                "steermoe": "OLMoE + SteerMoE (question only)",
            }
        )

    for case in plan.cases:
        case.full_prompt_text = case.evaluation_question
        if not case.prefix_conditioned_prompt_text:
            case.prefix_conditioned_prompt_text = build_concept_conditioned_evaluation_prompt(
                concept_type=plan.concept_type,
                concept_value=case.concept,
                evaluation_question=case.evaluation_question,
            )

    if args.limit_cases is not None:
        if args.limit_cases <= 0:
            raise ValueError("--limit-cases must be positive when provided.")
        plan.cases = plan.cases[: args.limit_cases]
        kept_concepts = {case.concept for case in plan.cases}
        kept_versions = {case.evaluation_version for case in plan.cases}
        plan.sampled_concepts = [
            concept for concept in plan.sampled_concepts if concept in kept_concepts
        ]
        plan.sampled_evaluation_versions = [
            version
            for version in plan.sampled_evaluation_versions
            if version in kept_versions
        ]
        plan.evaluation_questions_by_version = {
            version: question
            for version, question in plan.evaluation_questions_by_version.items()
            if version in kept_versions
        }

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

    resources = load_hf_model_resources(
        model_id=args.model_id,
        model_tag=args.model_tag,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
    )

    steermoe_plans_by_concept = {}
    for concept in plan.sampled_concepts:
        print(f"=== Building SteerMoE plan for {plan.concept_type}:{concept} ===")
        statement_prompt_pairs = build_upstream_statement_prompt_pairs(
            concept_type=plan.concept_type,
            concept_value=concept,
            general_statements_by_class=reference_data.general_statements_by_class,
            statement_stride=args.statement_stride,
        )
        custom_examples = build_custom_steering_examples_from_statement_prompt_pairs(
            statement_prompt_pairs
        )
        paired_traces = collect_olmoe_paired_routing_traces(
            examples=custom_examples,
            resources=resources,
            target_name=args.readout_target,
        )
        activation_table = build_steermoe_activation_table_from_paired_traces(
            paired_traces=paired_traces
        )
        steering_plan = build_steermoe_replication_plan(
            concept=concept,
            concept_type=plan.concept_type,
            model_id=resources.model_id,
            model_tag=resources.model_tag,
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
            dataset_dir / f"{resources.model_tag}_{concept}_custom_steering_dataset.json",
        )
        write_json(
            [asdict(trace) for trace in paired_traces],
            trace_dir / f"{resources.model_tag}_{concept}_routing_traces.json",
        )
        write_json(
            [asdict(score) for score in activation_table],
            activation_dir / f"{resources.model_tag}_{concept}_activation_table.json",
        )
        write_json(
            asdict(steering_plan),
            steering_dir / f"{resources.model_tag}_{concept}_steering_plan.json",
        )

    plan = fill_manual_review_plan_with_steermoe_generations(
        plan=plan,
        resources=resources,
        steermoe_plans_by_concept=steermoe_plans_by_concept,
        steering_coefficient=args.steering_coefficient,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    write_json(manual_review_plan_to_dict(plan), output_dir / "manual_review_plan.json")
    write_json(
        {
            "prompt_mode": "question_only",
            "prompt_contract": "baseline and SteerMoE receive only the evaluation question; the concept prefix is omitted at test time",
            "training_signal": "SteerMoE plans are still learned from prefix-conditioned vs unprefixed routing traces over matched statement bodies",
            "model_id": resources.model_id,
            "model_tag": resources.model_tag,
            "steering_coefficient": args.steering_coefficient,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "limit_cases": args.limit_cases,
        },
        output_dir / "generation_metadata.json",
    )
    (output_dir / "qualitative_review.md").write_text(build_manual_review_markdown(plan))
    (output_dir / "qualitative_review.html").write_text(
        build_manual_review_html(
            plan,
            title="OLMoE SteerMoE Qualitative Review",
        )
    )

    print(f"Wrote experiment bundle to {output_dir}")
    print(f"- {output_dir / 'manual_review_plan.json'}")
    print(f"- {output_dir / 'qualitative_review.md'}")
    print(f"- {output_dir / 'qualitative_review.html'}")


if __name__ == "__main__":
    main()
