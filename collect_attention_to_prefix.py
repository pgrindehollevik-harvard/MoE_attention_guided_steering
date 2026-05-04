from pathlib import Path
import argparse
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import (
    collect_attention_run,
    load_hf_model_resources,
    save_attention_run,
)
from moe_attention_guided_steering.reference_data import load_reference_data
from moe_attention_guided_steering.upstream_prompt_datasets import (
    build_upstream_statement_prompt_pairs,
)


def main() -> None:
    """Collect upstream-style attention-to-prefix scores on GPU-ready HF models."""
    parser = argparse.ArgumentParser(
        description="Collect per-layer attention-to-prefix scores for upstream-style concept prompts."
    )
    parser.add_argument(
        "--model-id",
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--model-tag",
        default="mixtral_8x7b_instruct_v0_1",
        help="Short filesystem-friendly name to use in output filenames.",
    )
    parser.add_argument(
        "--concept-type",
        default="fears",
        choices=[
            "custom",
            "fears",
            "jailbreaking",
            "moods",
            "personas",
            "personalities",
            "places",
        ],
        help="Concept family to collect attention traces for.",
    )
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Optional explicit concept list. If omitted, concepts are read from data/concepts.",
    )
    parser.add_argument(
        "--sample-concepts",
        type=int,
        default=None,
        help="Randomly sample this many concepts from the concept file.",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="Take the first N concepts after optional sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when sampling concepts.",
    )
    parser.add_argument(
        "--statement-stride",
        type=int,
        default=2,
        help="Stride over the combined statement list. Upstream attention collection uses 2 by default.",
    )
    parser.add_argument(
        "--head-aggregation",
        default="mean",
        choices=["mean", "max"],
        help="How to aggregate prefix-attention sums across heads.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the imported attention-guided steering data tree.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/attention_to_prefix",
        help="Directory where .npy and JSON attention artifacts should be written.",
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
        help="Transformers attention implementation. Upstream uses eager for attention extraction.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model with bitsandbytes 4-bit quantization when available.",
    )
    args = parser.parse_args()

    reference_data = load_reference_data(args.data_dir)
    if args.concepts:
        concepts = list(args.concepts)
    else:
        concepts = list(reference_data.concept_values_by_type[args.concept_type])

    if args.sample_concepts is not None:
        rng = random.Random(args.seed)
        concepts = rng.sample(concepts, args.sample_concepts)

    if args.max_concepts is not None:
        concepts = concepts[: args.max_concepts]

    resources = load_hf_model_resources(
        model_id=args.model_id,
        model_tag=args.model_tag,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
    )

    for concept in concepts:
        print(f"=== Collecting attention for {args.concept_type}:{concept} ===")
        prompt_pairs = build_upstream_statement_prompt_pairs(
            concept_type=args.concept_type,
            concept_value=concept,
            general_statements_by_class=reference_data.general_statements_by_class,
            statement_stride=args.statement_stride,
        )
        run = collect_attention_run(
            prompt_pairs=prompt_pairs,
            resources=resources,
            head_aggregation=args.head_aggregation,
            statement_stride=args.statement_stride,
        )
        paths = save_attention_run(run, args.output_dir)
        print(f"- array: {paths['array_path']}")
        print(f"- layer_to_token: {paths['layer_map_path']}")
        print(f"- metadata: {paths['metadata_path']}")


if __name__ == "__main__":
    main()
