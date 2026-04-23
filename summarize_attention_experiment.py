#!/usr/bin/env python3
"""Generate readable Markdown and HTML summaries for an attention experiment.

This script is meant for local inspection after a cluster run. It reads one
experiment directory organized like:

- experiments/<experiment_name>/<concept>/attention_scores.npy
- experiments/<experiment_name>/<concept>/metadata.json
- experiments/<experiment_name>/<concept>/layer_to_token.json

and produces two human-friendly reports in the experiment root:

- attention_report.md
- attention_report.html
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class LayerRow:
    """One layer's selected-token summary row."""

    layer_index: int
    selected_relative_index: int
    selected_token_text: str
    selected_mean_score: float
    selected_max_score: float
    top_statement_index: int
    top_statement_text: str


@dataclass
class PromptRow:
    """One prompt summary ranked by total selected attention."""

    rank: int
    prompt_id: str
    statement_index: int
    statement_text: str
    full_prompt_text: str
    total_selected_attention: float


@dataclass
class ConceptSummary:
    """All display-ready summary fields for one concept directory."""

    concept_name: str
    model_id: str
    head_aggregation: str
    array_shape: tuple[int, int, int]
    candidate_relative_indices: List[int]
    candidate_token_texts: List[str]
    token_usage_counts: Dict[int, int]
    token_global_mean_scores: Dict[int, float]
    layer_rows: List[LayerRow]
    prompt_rows: List[PromptRow]


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown and HTML summaries for one attention experiment directory."
    )
    parser.add_argument(
        "experiment_dir",
        help="Directory containing one subfolder per concept with .npy/.json artifacts.",
    )
    parser.add_argument(
        "--top-prompts",
        type=int,
        default=5,
        help="How many highest-scoring prompts to show per concept.",
    )
    return parser.parse_args()


def pretty_token_text(token_text: str) -> str:
    """Render token text in a way that keeps control characters readable."""
    return token_text.encode("unicode_escape").decode("ascii")


def load_concept_summary(concept_dir: Path, top_prompts: int) -> ConceptSummary:
    """Load one concept directory and compute readable statistics."""
    metadata = json.loads((concept_dir / "metadata.json").read_text())
    layer_to_token = {
        int(layer_index): int(relative_index)
        for layer_index, relative_index in json.loads(
            (concept_dir / "layer_to_token.json").read_text()
        ).items()
    }
    attention_array = np.load(concept_dir / "attention_scores.npy")

    candidate_relative_indices = list(metadata["candidate_token_relative_indices"])
    candidate_token_texts = list(metadata["candidate_token_texts"])
    relative_index_to_position = {
        relative_index: position
        for position, relative_index in enumerate(candidate_relative_indices)
    }

    token_usage_counts = Counter(layer_to_token.values())
    token_global_mean_scores = {
        relative_index: float(attention_array[:, :, position].mean())
        for position, relative_index in enumerate(candidate_relative_indices)
    }

    prompt_traces = metadata["prompt_traces"]
    selected_score_matrix = np.zeros(
        (attention_array.shape[0], attention_array.shape[1]),
        dtype=np.float32,
    )
    layer_rows: List[LayerRow] = []

    for layer_index in range(attention_array.shape[1]):
        selected_relative_index = layer_to_token[layer_index]
        selected_position = relative_index_to_position[selected_relative_index]
        selected_scores = attention_array[:, layer_index, selected_position]
        selected_score_matrix[:, layer_index] = selected_scores

        top_prompt_position = int(selected_scores.argmax())
        top_prompt_trace = prompt_traces[top_prompt_position]
        layer_rows.append(
            LayerRow(
                layer_index=layer_index,
                selected_relative_index=selected_relative_index,
                selected_token_text=pretty_token_text(
                    candidate_token_texts[selected_position]
                ),
                selected_mean_score=float(selected_scores.mean()),
                selected_max_score=float(selected_scores.max()),
                top_statement_index=int(top_prompt_trace["statement_index"]),
                top_statement_text=top_prompt_trace["statement_text"],
            )
        )

    prompt_total_scores = selected_score_matrix.sum(axis=1)
    top_prompt_positions = np.argsort(prompt_total_scores)[::-1][:top_prompts]
    prompt_rows: List[PromptRow] = []
    for rank, prompt_position in enumerate(top_prompt_positions, start=1):
        trace = prompt_traces[int(prompt_position)]
        prompt_rows.append(
            PromptRow(
                rank=rank,
                prompt_id=trace["prompt_id"],
                statement_index=int(trace["statement_index"]),
                statement_text=trace["statement_text"],
                full_prompt_text=trace["full_prompt_text"],
                total_selected_attention=float(prompt_total_scores[int(prompt_position)]),
            )
        )

    return ConceptSummary(
        concept_name=concept_dir.name,
        model_id=metadata["model_id"],
        head_aggregation=metadata["head_aggregation"],
        array_shape=tuple(int(value) for value in attention_array.shape),
        candidate_relative_indices=candidate_relative_indices,
        candidate_token_texts=[pretty_token_text(token) for token in candidate_token_texts],
        token_usage_counts=dict(sorted(token_usage_counts.items())),
        token_global_mean_scores=token_global_mean_scores,
        layer_rows=layer_rows,
        prompt_rows=prompt_rows,
    )


def render_candidate_list(summary: ConceptSummary) -> str:
    """Build a readable candidate-token summary list."""
    lines = []
    for relative_index, token_text in zip(
        summary.candidate_relative_indices,
        summary.candidate_token_texts,
    ):
        usage_count = summary.token_usage_counts.get(relative_index, 0)
        global_mean = summary.token_global_mean_scores[relative_index]
        lines.append(
            f"- `{relative_index}` -> `{token_text}`: chosen by {usage_count} / {summary.array_shape[1]} layers, "
            f"global mean score `{global_mean:.4f}`"
        )
    return "\n".join(lines)


def render_markdown(experiment_dir: Path, summaries: List[ConceptSummary]) -> str:
    """Render a Markdown report for the experiment."""
    first = summaries[0]
    lines = [
        "# Attention Experiment Report",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Model: `{first.model_id}`",
        f"- Head aggregation: `{first.head_aggregation}`",
        f"- Concepts: `{', '.join(summary.concept_name for summary in summaries)}`",
        "",
        "These artifacts are attention-selection outputs, not generated text responses.",
        "Each concept folder contains:",
        "- `attention_scores.npy`: raw attention scores with shape `(num_prompts, num_layers, num_candidate_tokens)`",
        "- `metadata.json`: prompt-level traces and token labels",
        "- `layer_to_token.json`: the final selected suffix token index for each layer",
        "",
    ]

    for summary in summaries:
        num_prompts, num_layers, num_candidates = summary.array_shape
        lines.extend(
            [
                f"## {summary.concept_name}",
                "",
                f"- Array shape: `{summary.array_shape}`",
                f"- Meaning: `{num_prompts}` prompts x `{num_layers}` layers x `{num_candidates}` candidate suffix tokens",
                "- Candidate token menu:",
                render_candidate_list(summary),
                "",
                "### Layer Selections",
                "",
                "| Layer | Selected Rel Idx | Token | Mean Selected Score | Max Selected Score | Highest-Scoring Statement |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in summary.layer_rows:
            escaped_statement = row.top_statement_text.replace("|", "\\|")
            lines.append(
                f"| {row.layer_index} | `{row.selected_relative_index}` | `{row.selected_token_text}` | "
                f"{row.selected_mean_score:.4f} | {row.selected_max_score:.4f} | "
                f"{escaped_statement} |"
            )

        lines.extend(
            [
                "",
                "### Top Prompt Traces",
                "",
            ]
        )
        for prompt_row in summary.prompt_rows:
            lines.extend(
                [
                    f"#### Rank {prompt_row.rank}",
                    "",
                    f"- Prompt id: `{prompt_row.prompt_id}`",
                    f"- Statement index: `{prompt_row.statement_index}`",
                    f"- Total selected attention: `{prompt_row.total_selected_attention:.4f}`",
                    f"- Statement text: {prompt_row.statement_text}",
                    "",
                    "```text",
                    prompt_row.full_prompt_text,
                    "```",
                    "",
                ]
            )

    return "\n".join(lines)


def render_html(experiment_dir: Path, summaries: List[ConceptSummary]) -> str:
    """Render a self-contained HTML report for the experiment."""
    first = summaries[0]
    overview_items = "".join(
        f"<li><strong>{html.escape(summary.concept_name)}</strong>: shape {summary.array_shape}</li>"
        for summary in summaries
    )

    concept_blocks: List[str] = []
    for summary in summaries:
        candidate_items = "".join(
            (
                "<li>"
                f"<code>{relative_index}</code> -> <code>{html.escape(token_text)}</code>"
                f" <span class='muted'>(chosen by {summary.token_usage_counts.get(relative_index, 0)} / {summary.array_shape[1]} "
                f"layers, global mean {summary.token_global_mean_scores[relative_index]:.4f})</span>"
                "</li>"
            )
            for relative_index, token_text in zip(
                summary.candidate_relative_indices,
                summary.candidate_token_texts,
            )
        )

        layer_rows_html = "".join(
            (
                "<tr>"
                f"<td>{row.layer_index}</td>"
                f"<td><code>{row.selected_relative_index}</code></td>"
                f"<td><code>{html.escape(row.selected_token_text)}</code></td>"
                f"<td>{row.selected_mean_score:.4f}</td>"
                f"<td>{row.selected_max_score:.4f}</td>"
                f"<td>{html.escape(row.top_statement_text)}</td>"
                "</tr>"
            )
            for row in summary.layer_rows
        )

        prompt_cards = "".join(
            (
                "<details class='prompt-card'>"
                f"<summary>Rank {prompt_row.rank}: statement {prompt_row.statement_index} "
                f"(total selected attention {prompt_row.total_selected_attention:.4f})</summary>"
                f"<p><strong>Prompt id:</strong> <code>{html.escape(prompt_row.prompt_id)}</code></p>"
                f"<p><strong>Statement text:</strong> {html.escape(prompt_row.statement_text)}</p>"
                f"<pre>{html.escape(prompt_row.full_prompt_text)}</pre>"
                "</details>"
            )
            for prompt_row in summary.prompt_rows
        )

        concept_blocks.append(
            f"""
            <section class="concept">
              <h2>{html.escape(summary.concept_name)}</h2>
              <p class="muted">
                shape {summary.array_shape} = {summary.array_shape[0]} prompts x
                {summary.array_shape[1]} layers x {summary.array_shape[2]} candidate suffix tokens
              </p>

              <h3>Candidate Token Menu</h3>
              <ul>{candidate_items}</ul>

              <h3>Layer Selections</h3>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th>Selected Rel Idx</th>
                      <th>Token</th>
                      <th>Mean Selected Score</th>
                      <th>Max Selected Score</th>
                      <th>Highest-Scoring Statement</th>
                    </tr>
                  </thead>
                  <tbody>{layer_rows_html}</tbody>
                </table>
              </div>

              <h3>Top Prompt Traces</h3>
              {prompt_cards}
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attention Experiment Report</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --paper: #fffdf9;
      --ink: #1f1a17;
      --muted: #655d57;
      --line: #d7cbc0;
      --accent: #9d5c2f;
      --accent-soft: #efe0d3;
      --code: #f4eee7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: linear-gradient(180deg, #efe6d9 0%, var(--bg) 35%, #f8f4ee 100%);
      color: var(--ink);
      line-height: 1.5;
    }}
    main {{
      width: min(1180px, calc(100vw - 48px));
      margin: 32px auto 64px;
      background: color-mix(in srgb, var(--paper) 94%, white 6%);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(76, 48, 22, 0.08);
      overflow: hidden;
    }}
    header {{
      padding: 28px 32px;
      background: linear-gradient(135deg, #f8efe2 0%, #f3e3cf 100%);
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3 {{
      font-family: "Avenir Next Condensed", "Gill Sans", sans-serif;
      letter-spacing: 0.02em;
      margin: 0 0 10px;
    }}
    h1 {{ font-size: 2.4rem; }}
    h2 {{
      font-size: 1.8rem;
      padding-top: 6px;
    }}
    h3 {{
      margin-top: 22px;
      font-size: 1.15rem;
      color: var(--accent);
    }}
    p, li {{ font-size: 1rem; }}
    code {{
      font-family: "SF Mono", "Menlo", monospace;
      background: var(--code);
      padding: 0.12rem 0.35rem;
      border-radius: 6px;
      font-size: 0.92em;
    }}
    pre {{
      white-space: pre-wrap;
      background: var(--code);
      padding: 14px 16px;
      border-radius: 12px;
      border: 1px solid var(--line);
      overflow-x: auto;
    }}
    .muted {{ color: var(--muted); }}
    .content {{
      padding: 28px 32px 40px;
    }}
    .overview {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 24px;
      align-items: start;
      margin-bottom: 24px;
    }}
    .overview-card {{
      background: #fffaf3;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 20px;
    }}
    .concept {{
      padding: 28px 0;
      border-top: 1px solid var(--line);
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #ece3d8;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f9f0e4;
      z-index: 1;
    }}
    .prompt-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 16px;
      background: #fffaf5;
      margin-bottom: 12px;
    }}
    .prompt-card summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      .overview {{
        grid-template-columns: 1fr;
      }}
      main {{
        width: min(100vw - 20px, 1180px);
        margin: 10px auto 24px;
      }}
      header, .content {{
        padding: 20px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Attention Experiment Report</h1>
      <p class="muted">
        Local summary for <code>{html.escape(str(experiment_dir))}</code>
      </p>
    </header>
    <div class="content">
      <section class="overview">
        <div class="overview-card">
          <h2>Overview</h2>
          <p><strong>Model:</strong> <code>{html.escape(first.model_id)}</code></p>
          <p><strong>Head aggregation:</strong> <code>{html.escape(first.head_aggregation)}</code></p>
          <p class="muted">
            These are attention-selection artifacts, not generated model answers.
            Each concept folder contains raw scores, metadata traces, and one final
            layer-to-token selection map.
          </p>
        </div>
        <div class="overview-card">
          <h2>Concepts</h2>
          <ul>{overview_items}</ul>
        </div>
      </section>
      {''.join(concept_blocks)}
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    """Load one experiment directory and write Markdown and HTML reports."""
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    concept_dirs = sorted(path for path in experiment_dir.iterdir() if path.is_dir())
    if not concept_dirs:
        raise FileNotFoundError(f"No concept directories found under {experiment_dir}")

    summaries = [
        load_concept_summary(concept_dir=concept_dir, top_prompts=args.top_prompts)
        for concept_dir in concept_dirs
    ]

    markdown_path = experiment_dir / "attention_report.md"
    html_path = experiment_dir / "attention_report.html"

    markdown_path.write_text(render_markdown(experiment_dir, summaries))
    html_path.write_text(render_html(experiment_dir, summaries))

    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
