"""Aggregate experiment artifacts into paper-friendly summary tables.

This script walks `experiments/` and emits two CSVs that the paper
notebook and figure code consume:

    paper/notebooks/sweep_results.csv
        One row per (run, concept) capturing the steering knobs and a
        light qualitative shape of the responses (length, lexical overlap
        with the concept, etc.). Designed to mirror the role
        ``sweep_results.csv`` plays in the hyperbolic submission.

    paper/notebooks/expert_selection.csv
        One row per (run, concept, layer, expert) capturing the
        risk-difference activation table that drove the SteerMoE plan.

Pure-Python with a `pandas` fallback. Reads only the metadata, manual
review plans, qualitative review markdown, and activation tables that
are already committed under `experiments/`. Does not load any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
DEFAULT_SWEEP_CSV = REPO_ROOT / "paper" / "notebooks" / "sweep_results.csv"
DEFAULT_EXPERT_CSV = REPO_ROOT / "paper" / "notebooks" / "expert_selection.csv"

QUESTION_ONLY_PREFIX = "mixtral_steermoe_fears_seed7_question_only_"
PREFIX_CONDITIONED_PREFIX = "mixtral_steermoe_fears_seed7_prefix_conditioned_"


@dataclass
class RunMetadata:
    run_dir: Path
    metadata: dict
    plan: dict
    response_blocks: list[dict] = field(default_factory=list)


def _read_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "generation_metadata.json").exists():
            yield entry


CONCEPT_HEADER_RE = re.compile(r"^## (?P<concept>[^\n#]+?)\s*$")
EVAL_HEADER_RE = re.compile(r"^### Eval v(?P<version>\d+)\s*$")
QUESTION_RE = re.compile(r"^- Question: (?P<text>.+)$")
RESPONSE_RE = re.compile(r"^- (?P<label>.+) response:\s*$")


def _parse_qualitative_review(md_path: Path) -> list[dict]:
    """Parse the manual review markdown into structured response blocks.

    Returns a list of dicts, one per (concept, eval_version, condition).
    Falls back gracefully when the markdown predates the current schema.
    """

    if not md_path.exists():
        return []

    blocks: list[dict] = []
    current_concept: str | None = None
    current_version: int | None = None
    current_question: str | None = None
    current_label: str | None = None
    response_lines: list[str] = []

    def _flush_response() -> None:
        nonlocal response_lines, current_label
        if current_label is not None and current_concept is not None:
            blocks.append(
                {
                    "concept": current_concept,
                    "evaluation_version": current_version,
                    "question": current_question,
                    "condition_label": current_label,
                    "response": "\n".join(response_lines).strip(),
                }
            )
        response_lines = []
        current_label = None

    with md_path.open() as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if current_label is not None:
                if line.startswith("- ") or line.startswith("### ") or line.startswith("## "):
                    _flush_response()
                else:
                    response_lines.append(line)
                    continue

            concept_match = CONCEPT_HEADER_RE.match(line)
            if concept_match:
                current_concept = concept_match.group("concept").strip()
                current_version = None
                current_question = None
                continue

            eval_match = EVAL_HEADER_RE.match(line)
            if eval_match:
                current_version = int(eval_match.group("version"))
                continue

            question_match = QUESTION_RE.match(line)
            if question_match:
                current_question = question_match.group("text").strip()
                continue

            response_match = RESPONSE_RE.match(line)
            if response_match:
                current_label = response_match.group("label").strip()
                response_lines = []
                continue

        _flush_response()

    return blocks


def _summarise_response(text: str, concept: str) -> dict[str, float]:
    if not text:
        return {"word_count": 0, "concept_token_hit": 0, "exceeds_20_words": 0}
    words = re.findall(r"[A-Za-z']+", text)
    concept_tokens = {tok.lower() for tok in re.findall(r"[A-Za-z]+", concept) if len(tok) > 2}
    lower_words = [w.lower() for w in words]
    overlap = sum(1 for w in lower_words if w in concept_tokens)
    return {
        "word_count": len(words),
        "concept_token_hit": int(overlap > 0),
        "exceeds_20_words": int(len(words) > 20),
    }


def _classify_condition(label: str) -> str:
    lower = label.lower()
    if "+ steermoe" in lower or "steermoe" in lower:
        return "mixtral_steermoe"
    if "mixtral" in lower:
        return "mixtral_baseline"
    if "llama" in lower:
        return "llama_baseline"
    if "olmoe" in lower:
        return "olmoe_baseline"
    return "other"


def _load_runs() -> list[RunMetadata]:
    runs: list[RunMetadata] = []
    for run_dir in _iter_run_dirs(EXPERIMENTS):
        meta = _read_json(run_dir / "generation_metadata.json")
        plan_path = run_dir / "manual_review_plan.json"
        plan = _read_json(plan_path) if plan_path.exists() else {}
        review = _parse_qualitative_review(run_dir / "qualitative_review.md")
        runs.append(RunMetadata(run_dir=run_dir, metadata=meta, plan=plan, response_blocks=review))
    return runs


def _run_kind(run_dir_name: str) -> str:
    if run_dir_name.startswith(PREFIX_CONDITIONED_PREFIX):
        return "prefix_conditioned"
    if run_dir_name.startswith(QUESTION_ONLY_PREFIX):
        return "question_only"
    if "prefix_conditioned" in run_dir_name:
        return "prefix_conditioned"
    if "question_only" in run_dir_name:
        return "question_only"
    return "unknown"


def _emit_sweep_rows(runs: list[RunMetadata]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        meta = run.metadata
        kind = _run_kind(run.run_dir.name)
        for block in run.response_blocks:
            shape = _summarise_response(block["response"], block["concept"] or "")
            rows.append(
                {
                    "run_name": run.run_dir.name,
                    "run_kind": kind,
                    "prompt_mode": meta.get("prompt_mode"),
                    "steering_coefficient": meta.get("steering_coefficient"),
                    "top_positive_experts": meta.get("top_positive_experts"),
                    "top_negative_experts": meta.get("top_negative_experts"),
                    "minimum_abs_risk_difference": meta.get("minimum_abs_risk_difference"),
                    "include_llama_reference": meta.get("include_llama_reference"),
                    "concept": block["concept"],
                    "evaluation_version": block["evaluation_version"],
                    "condition_label": block["condition_label"],
                    "condition_kind": _classify_condition(block["condition_label"]),
                    "response": block["response"],
                    **shape,
                }
            )
    return rows


def _emit_expert_rows(runs: list[RunMetadata]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        tables_dir = run.run_dir / "activation_tables"
        if not tables_dir.exists():
            continue
        meta = run.metadata
        for table_path in sorted(tables_dir.glob("*.json")):
            stem = table_path.stem
            concept = stem.split("_activation_table")[0]
            payload = _read_json(table_path)
            if isinstance(payload, dict):
                entries = payload.get("activation_table") or []
            else:
                entries = payload
            for row in entries:
                rows.append(
                    {
                        "run_name": run.run_dir.name,
                        "concept": concept,
                        "layer_index": row["layer_index"],
                        "expert_index": row["expert_index"],
                        "risk_difference": row["risk_difference"],
                        "abs_risk_difference": row["abs_risk_difference"],
                        "messages_0_activation_rate": row.get("messages_0_activation_rate"),
                        "messages_1_activation_rate": row.get("messages_1_activation_rate"),
                        "messages_0_token_count": row.get("messages_0_token_count"),
                        "messages_1_token_count": row.get("messages_1_token_count"),
                        "steering_coefficient": meta.get("steering_coefficient"),
                        "top_positive_experts": meta.get("top_positive_experts"),
                        "top_negative_experts": meta.get("top_negative_experts"),
                    }
                )
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--expert-csv", type=Path, default=DEFAULT_EXPERT_CSV)
    args = parser.parse_args()

    runs = _load_runs()
    sweep_rows = _emit_sweep_rows(runs)
    expert_rows = _emit_expert_rows(runs)

    _write_csv(sweep_rows, args.sweep_csv)
    _write_csv(expert_rows, args.expert_csv)

    print(f"runs: {len(runs)}")
    print(f"sweep rows: {len(sweep_rows)} -> {args.sweep_csv.relative_to(REPO_ROOT)}")
    print(f"expert rows: {len(expert_rows)} -> {args.expert_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
