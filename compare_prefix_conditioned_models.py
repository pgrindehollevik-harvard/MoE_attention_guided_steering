#!/usr/bin/env python3
"""Compare unsteered models on explicit prefix-conditioned evaluation prompts."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.io_utils import write_json  # noqa: E402
from moe_attention_guided_steering.manual_review import load_manual_review_plan  # noqa: E402
from moe_attention_guided_steering.model_comparison import (  # noqa: E402
    build_empty_model_comparison_results,
    fill_model_comparison_results_with_generations,
    load_model_comparison_specs,
    render_model_comparison_html,
    render_model_comparison_markdown,
)


def main() -> None:
    """Run the full-prefix model suitability diagnostic requested by Parmida."""
    parser = argparse.ArgumentParser(
        description="Compare unsteered model behavior when the explicit concept prefix is included."
    )
    parser.add_argument(
        "--plan-json",
        default="outputs/manual_fear_review/manual_review_plan.json",
        help="Manual review plan JSON describing concepts/questions to run.",
    )
    parser.add_argument(
        "--model-config-json",
        default="configs/prefix_conditioned_model_comparison.json",
        help="JSON file listing model ids, labels, prompt formats, and trust_remote_code settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/prefix_conditioned_model_comparison_fears_seed7",
        help="Experiment directory where comparison JSON, Markdown, and HTML should be written.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["prefix", "question"],
        default="prefix",
        help="Use prefix+question for model suitability, or question-only for a control diagnostic.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of concept/question cases.",
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
        default=None,
        help="Optional global Transformers device_map override. Omit to use per-model config.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype name passed to from_pretrained, such as bfloat16 or float16.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional global override for Transformers attn_implementation. Omit to use per-model config.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load models with bitsandbytes 4-bit quantization when available.",
    )
    parser.add_argument(
        "--bnb-cpu-offload",
        action="store_true",
        help="Allow bitsandbytes to keep overflow modules on CPU when a quantized model does not fit GPU RAM.",
    )
    args = parser.parse_args()

    plan = load_manual_review_plan(args.plan_json)
    if args.limit_cases is not None:
        plan.cases = plan.cases[: args.limit_cases]

    model_specs = load_model_comparison_specs(args.model_config_json)
    results = build_empty_model_comparison_results(
        plan=plan,
        model_specs=model_specs,
        prompt_mode=args.prompt_mode,
    )
    results = fill_model_comparison_results_with_generations(
        results=results,
        model_specs=model_specs,
        cache_dir=args.cache_dir,
        device_map_override=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation_override=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        bnb_cpu_offload=args.bnb_cpu_offload,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(results, output_dir / "model_comparison_results.json")
    (output_dir / "model_comparison.md").write_text(
        render_model_comparison_markdown(results)
    )
    (output_dir / "model_comparison.html").write_text(
        render_model_comparison_html(results)
    )


if __name__ == "__main__":
    main()
