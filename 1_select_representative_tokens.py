from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from args import build_parser
from moe_attention_guided_steering.attention_utils import index_selected_tokens_by_layer_and_label
from moe_attention_guided_steering.datasets import load_experiment
from moe_attention_guided_steering.io_utils import ensure_directory, write_json


def main() -> None:
    """Select representative tokens and save them as JSON.

    We keep this stage as a standalone script because token selection is one of the
    most important design decisions in the hybrid method, and it is useful to be
    able to inspect its outputs independently from the MoE scoring stage.
    """
    parser = build_parser("Select representative tokens for each example and layer.")
    args = parser.parse_args()

    dataset = load_experiment(args.input)
    selected = index_selected_tokens_by_layer_and_label(
        dataset.examples,
        minimum_attention_weight=args.min_attention,
    )
    output_dir = ensure_directory(args.output_dir)
    payload = {
        str(layer_index): {
            label: [token.__dict__ for token in tokens]
            for label, tokens in label_map.items()
        }
        for layer_index, label_map in selected.items()
    }
    output_path = output_dir / "selected_tokens.json"
    write_json(payload, output_path)
    print(f"Wrote representative tokens to {output_path}")


if __name__ == "__main__":
    main()
