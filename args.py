import argparse


def build_parser(description: str, include_intervention_args: bool = False) -> argparse.ArgumentParser:
    """Build a shared CLI parser used across the numbered scripts.

    We keep argument creation centralized for the same reason the attention-guided
    reference repo does: researchers typically run the same experiment pipeline
    in several stages, and duplicated CLI definitions drift over time.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        default="examples/toy_experiment.json",
        help="Path to the JSON file containing attention traces and expert loads.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/demo",
        help="Directory where JSON artifacts and reports should be written.",
    )
    parser.add_argument(
        "--min-attention",
        type=float,
        default=0.0,
        help="Discard token candidates below this attention weight before selection.",
    )

    if include_intervention_args:
        parser.add_argument(
            "--top-k-experts",
            type=int,
            default=1,
            help="Maximum number of experts to activate and deactivate per layer.",
        )
        parser.add_argument(
            "--activation-threshold",
            type=float,
            default=0.2,
            help="Minimum positive delta required to recommend expert activation.",
        )
        parser.add_argument(
            "--deactivation-threshold",
            type=float,
            default=-0.2,
            help="Maximum negative delta required to recommend expert deactivation.",
        )

    return parser
