#!/usr/bin/env python3
"""Run the first end-to-end OLMoE qualitative review experiment."""

from dataclasses import asdict
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import (  # noqa: E402
    collect_attention_run,
    load_hf_model_resources,
    save_attention_run,
)
from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig  # noqa: E402
from moe_attention_guided_steering.io_utils import write_json  # noqa: E402
from moe_attention_guided_steering.manual_review import (  # noqa: E402
    build_manual_review_html,
    build_manual_review_markdown,
    load_manual_review_plan,
    manual_review_plan_to_dict,
)
from moe_attention_guided_steering.olmoe_backend import (  # noqa: E402
    build_fixed_layer_to_token_index,
    build_experiment_dataset_from_router_traces,
    collect_olmoe_router_traces_for_prompt_pairs,
    fill_manual_review_plan_with_olmoe_generations,
    load_layer_to_token_index,
    run_olmoe_steering_pipeline,
)
from moe_attention_guided_steering.reference_data import load_reference_data  # noqa: E402
from moe_attention_guided_steering.upstream_prompt_datasets import (  # noqa: E402
    build_upstream_statement_prompt_pairs,
)


def main() -> None:
    """Build OLMoE steering plans and fill the three-way manual review bundle.

    High-level stages:
    1. Load the sampled manual review cases.
    2. For each concept in the plan, collect or reuse attention-guided token maps.
    3. Collect OLMoE router traces and build two steering plans:
       - original MoESteer using one fixed suffix token,
       - attention-guided MoESteer using per-layer attention-selected tokens.
    4. Generate qualitative responses for:
       - baseline,
       - original MoESteer,
       - attention-guided MoESteer.
    5. Write the filled JSON plan plus browsable HTML/Markdown reports.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline vs MoESteer vs attention-guided MoESteer on OLMoE and render a qualitative review page."
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
        default="experiments/olmoe_fears_seed7",
        help="Experiment directory where attention artifacts, steering plans, and reports should be written.",
    )
    parser.add_argument(
        "--statement-stride",
        type=int,
        default=2,
        help="Stride over the upstream statement pool when building concept prompt pairs.",
    )
    parser.add_argument(
        "--head-aggregation",
        default="mean",
        choices=["mean", "max"],
        help="How to aggregate attention-to-prefix scores across heads.",
    )
    parser.add_argument(
        "--fixed-token-index",
        type=int,
        default=-1,
        help="Original MoESteer token-choice baseline. -1 means the final shared suffix token.",
    )
    parser.add_argument(
        "--top-k-experts",
        type=int,
        default=4,
        help="Maximum number of experts to activate and deactivate in each layer.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.002,
        help="Minimum probability delta required to activate an expert.",
    )
    parser.add_argument(
        "--deactivation-threshold",
        type=float,
        default=-0.002,
        help="Maximum probability delta required to deactivate an expert.",
    )
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=8.0,
        help="Dense router-logit bias added to selected experts during generation.",
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
        help="Transformers attention implementation. Eager is safest when attention outputs are needed.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load OLMoE with bitsandbytes 4-bit quantization when available.",
    )
    args = parser.parse_args()

    plan = load_manual_review_plan(args.plan_json)
    reference_data = load_reference_data(args.data_dir)
    output_dir = Path(args.output_dir)
    attention_dir = output_dir / "attention_to_prefix"
    dataset_dir = output_dir / "router_datasets"
    steering_dir = output_dir / "steering_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)
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

    moesteer_plans_by_concept = {}
    attention_guided_plans_by_concept = {}

    for concept in plan.sampled_concepts:
        print(f"=== Building steering plans for {plan.concept_type}:{concept} ===")
        prompt_pairs = build_upstream_statement_prompt_pairs(
            concept_type=plan.concept_type,
            concept_value=concept,
            general_statements_by_class=reference_data.general_statements_by_class,
            statement_stride=args.statement_stride,
        )

        layer_map_path = (
            attention_dir
            / f"attentions_{args.head_aggregation}head_{resources.model_tag}_{concept}_paired_statements.layer_to_token.json"
        )
        if layer_map_path.exists():
            attention_layer_to_token = load_layer_to_token_index(str(layer_map_path))
        else:
            attention_run = collect_attention_run(
                prompt_pairs=prompt_pairs,
                resources=resources,
                head_aggregation=args.head_aggregation,
                statement_stride=args.statement_stride,
            )
            save_attention_run(attention_run, str(attention_dir))
            attention_layer_to_token = attention_run.layer_to_token_index

        fixed_layer_to_token = build_fixed_layer_to_token_index(
            num_layers=len(attention_layer_to_token),
            fixed_token_index=args.fixed_token_index,
        )

        router_traces = collect_olmoe_router_traces_for_prompt_pairs(
            prompt_pairs=prompt_pairs,
            resources=resources,
        )
        fixed_dataset = build_experiment_dataset_from_router_traces(
            prompt_traces=router_traces,
            concept=concept,
            contrast_concept="generic_statement_control",
            concept_type=plan.concept_type,
            model_id=resources.model_id,
            model_tag=resources.model_tag,
            selection_layer_to_token_index=fixed_layer_to_token,
            selection_source=f"fixed_token_{args.fixed_token_index}",
        )
        attention_dataset = build_experiment_dataset_from_router_traces(
            prompt_traces=router_traces,
            concept=concept,
            contrast_concept="generic_statement_control",
            concept_type=plan.concept_type,
            model_id=resources.model_id,
            model_tag=resources.model_tag,
            selection_layer_to_token_index=attention_layer_to_token,
            selection_source="attention_guided",
        )

        fixed_artifacts = run_olmoe_steering_pipeline(fixed_dataset, config=config)
        attention_artifacts = run_olmoe_steering_pipeline(attention_dataset, config=config)

        write_json(
            asdict(fixed_dataset),
            dataset_dir / f"{resources.model_tag}_{concept}_fixed_dataset.json",
        )
        write_json(
            asdict(attention_dataset),
            dataset_dir / f"{resources.model_tag}_{concept}_attention_guided_dataset.json",
        )
        write_json(
            asdict(fixed_artifacts.steering_plan),
            steering_dir / f"{resources.model_tag}_{concept}_fixed_steering_plan.json",
        )
        write_json(
            asdict(attention_artifacts.steering_plan),
            steering_dir / f"{resources.model_tag}_{concept}_attention_guided_steering_plan.json",
        )

        moesteer_plans_by_concept[concept] = fixed_artifacts.steering_plan
        attention_guided_plans_by_concept[concept] = attention_artifacts.steering_plan

    fill_manual_review_plan_with_olmoe_generations(
        plan=plan,
        resources=resources,
        moesteer_plans_by_concept=moesteer_plans_by_concept,
        attention_guided_plans_by_concept=attention_guided_plans_by_concept,
        steering_coefficient=args.steering_coefficient,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    plan_json_path = output_dir / "manual_review_plan.json"
    markdown_path = output_dir / "qualitative_review.md"
    html_path = output_dir / "qualitative_review.html"
    attention_report_path = attention_dir / "attention_report.html"

    write_json(manual_review_plan_to_dict(plan), plan_json_path)
    markdown_path.write_text(build_manual_review_markdown(plan))
    html_path.write_text(
        build_manual_review_html(
            plan,
            title=f"Qualitative {plan.concept_type.title()} Comparison ({resources.model_tag})",
            companion_attention_report=str(attention_report_path) if attention_report_path.exists() else "",
        )
    )

    print(f"Wrote {plan_json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
