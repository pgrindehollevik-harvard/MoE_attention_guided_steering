#!/usr/bin/env python3
"""Append a question-only unsteered reference-model column to a review report."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import load_hf_model_resources  # noqa: E402
from moe_attention_guided_steering.generation import generate_unsteered_response  # noqa: E402
from moe_attention_guided_steering.io_utils import write_json  # noqa: E402
from moe_attention_guided_steering.manual_review import (  # noqa: E402
    build_manual_review_html,
    build_manual_review_markdown,
    load_manual_review_plan,
    manual_review_plan_to_dict,
)


def main() -> None:
    """Fill an existing question-only steering report with a dense reference model."""
    parser = argparse.ArgumentParser(
        description="Add a question-only unsteered reference model column to an existing manual review plan."
    )
    parser.add_argument(
        "--plan-json",
        default="experiments/olmoe_steermoe_fears_seed7_question_only/manual_review_plan.json",
        help="Existing manual review plan JSON from the question-only SteerMoE run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the updated plan and reports should be written. Defaults to the plan JSON directory.",
    )
    parser.add_argument(
        "--condition-key",
        default="llama_3_1_8b_baseline",
        help="Stable response key for the appended reference column.",
    )
    parser.add_argument(
        "--condition-label",
        default="Llama 3.1 8B baseline (question only)",
        help="Human-readable label for the appended reference column.",
    )
    parser.add_argument(
        "--insert-after",
        default="baseline",
        help="Existing condition key after which the reference column should be inserted.",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id for the unsteered reference model.",
    )
    parser.add_argument(
        "--model-tag",
        default="llama_3_1_8b_instruct",
        help="Filesystem-friendly model tag for metadata.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map argument for the reference model.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype name passed to from_pretrained.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the reference model with bitsandbytes 4-bit quantization when available.",
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
    args = parser.parse_args()

    plan_json = Path(args.plan_json)
    output_dir = Path(args.output_dir) if args.output_dir else plan_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = load_manual_review_plan(plan_json)
    if args.condition_key not in plan.condition_order:
        if args.insert_after in plan.condition_order:
            insert_at = plan.condition_order.index(args.insert_after) + 1
            plan.condition_order = [
                *plan.condition_order[:insert_at],
                args.condition_key,
                *plan.condition_order[insert_at:],
            ]
        else:
            plan.condition_order = [args.condition_key, *plan.condition_order]
    plan.condition_labels[args.condition_key] = args.condition_label
    for case in plan.cases:
        case.responses.setdefault(args.condition_key, "")

    resources = load_hf_model_resources(
        model_id=args.model_id,
        model_tag=args.model_tag,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
        infer_attention_suffix_tokens=False,
    )

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    response_cache = {}
    for case in tqdm(plan.cases, desc=f"Generating {args.model_tag} reference responses"):
        prompt_text = case.full_prompt_text.strip() or case.evaluation_question
        if prompt_text not in response_cache:
            response_cache[prompt_text] = generate_unsteered_response(
                prompt_text=prompt_text,
                resources=resources,
                prompt_format="chat",
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        case.responses[args.condition_key] = response_cache[prompt_text]

    write_json(
        {
            "prompt_mode": "question_only",
            "prompt_contract": "reference model receives only the evaluation question; no concept prefix is used",
            "model_id": resources.model_id,
            "model_tag": resources.model_tag,
            "condition_key": args.condition_key,
            "condition_label": args.condition_label,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        output_dir / "reference_generation_metadata.json",
    )
    write_json(manual_review_plan_to_dict(plan), output_dir / "manual_review_plan.json")
    (output_dir / "qualitative_review.md").write_text(build_manual_review_markdown(plan))
    (output_dir / "qualitative_review.html").write_text(
        build_manual_review_html(
            plan,
            title="Question-Only Steering Review With Llama Reference",
        )
    )


if __name__ == "__main__":
    main()
