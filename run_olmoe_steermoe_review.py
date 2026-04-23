#!/usr/bin/env python3
"""Run a baseline-vs-SteerMoE qualitative review experiment on OLMoE."""

from dataclasses import asdict
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig  # noqa: E402
from moe_attention_guided_steering.io_utils import write_json  # noqa: E402
from moe_attention_guided_steering.manual_review import (  # noqa: E402
    build_manual_review_html,
    build_manual_review_markdown,
    load_manual_review_plan,
    manual_review_plan_to_dict,
)
from moe_attention_guided_steering.olmoe_backend import (  # noqa: E402
    collect_olmoe_experiment_dataset_for_prompt_pairs,
    fill_manual_review_plan_with_steermoe_generations,
    run_olmoe_steermoe_pipeline,
)
from moe_attention_guided_steering.reference_data import load_reference_data  # noqa: E402
from moe_attention_guided_steering.attention_collection import load_hf_model_resources  # noqa: E402
from moe_attention_guided_steering.upstream_prompt_datasets import (  # noqa: E402
    build_upstream_statement_prompt_pairs,
)


def main() -> None:
    """Run the stage-1 OLMoE transfer experiment: baseline vs SteerMoE.

    High-level stages:
    1. Load the sampled manual review cases.
    2. For each concept, build upstream-style positive/negative statement pairs.
    3. Collect OLMoE router traces over the full user-content span.
    4. Aggregate those traces into span-level expert activation rates.
    5. Build one sparse SteerMoE plan per concept.
    6. Generate baseline and SteerMoE answers into the review bundle.
    7. Write the filled JSON plan plus browsable HTML/Markdown reports.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline vs faithful-ish span-based SteerMoE on OLMoE and render a qualitative review page."
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
        default="experiments/olmoe_steermoe_fears_seed7",
        help="Experiment directory where datasets, steering plans, and reports should be written.",
    )
    parser.add_argument(
        "--statement-stride",
        type=int,
        default=2,
        help="Stride over the upstream statement pool when building concept prompt pairs.",
    )
    parser.add_argument(
        "--readout-span",
        default="user_content",
        choices=["user_content"],
        help="Which prompt span to aggregate when computing SteerMoE routing statistics.",
    )
    parser.add_argument(
        "--top-k-experts",
        type=int,
        default=2,
        help="Maximum number of experts to activate and deactivate in each layer.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.01,
        help="Minimum activation-rate delta required to activate an expert.",
    )
    parser.add_argument(
        "--deactivation-threshold",
        type=float,
        default=-0.01,
        help="Maximum activation-rate delta required to deactivate an expert.",
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
        help="Load OLMoE with bitsandbytes 4-bit quantization when available.",
    )
    args = parser.parse_args()

    plan = load_manual_review_plan(args.plan_json)
    if plan.condition_order != ["baseline", "steermoe"]:
        plan.condition_order = ["baseline", "steermoe"]
        plan.condition_labels = {
            "baseline": "Baseline (no steering)",
            "steermoe": "SteerMoE",
        }
        for case in plan.cases:
            baseline_value = case.responses.get("baseline", "")
            steermoe_value = case.responses.get("steermoe", "")
            case.responses = {
                "baseline": baseline_value,
                "steermoe": steermoe_value,
            }

    reference_data = load_reference_data(args.data_dir)
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "router_datasets"
    steering_dir = output_dir / "steering_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
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

    config = ExperimentConfig(
        intervention=InterventionConfig(
            top_k_experts=args.top_k_experts,
            activation_threshold=args.activation_threshold,
            deactivation_threshold=args.deactivation_threshold,
        )
    )

    steermoe_plans_by_concept = {}
    for concept in plan.sampled_concepts:
        print(f"=== Building SteerMoE plan for {plan.concept_type}:{concept} ===")
        prompt_pairs = build_upstream_statement_prompt_pairs(
            concept_type=plan.concept_type,
            concept_value=concept,
            general_statements_by_class=reference_data.general_statements_by_class,
            statement_stride=args.statement_stride,
        )

        dataset = collect_olmoe_experiment_dataset_for_prompt_pairs(
            prompt_pairs=prompt_pairs,
            resources=resources,
            readout_span=args.readout_span,
        )
        dataset.metadata["statement_stride"] = str(args.statement_stride)
        dataset.metadata["readout_span"] = args.readout_span
        artifacts = run_olmoe_steermoe_pipeline(dataset, config=config)
        steermoe_plans_by_concept[concept] = artifacts.steering_plan

        write_json(
            asdict(dataset),
            dataset_dir / f"{resources.model_tag}_{concept}_steermoe_dataset.json",
        )
        write_json(
            asdict(artifacts.steering_plan),
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
