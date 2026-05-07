#!/usr/bin/env python3
"""Compile Mixtral question-only steering ablations into one HTML report.

The report is intentionally built from saved experiment bundles. It does not
rerun any model code. Each run directory is expected to contain:

- manual_review_plan.json
- generation_metadata.json

The default run set compares the original Mixtral SteerMoE result against the
small ablation grid used to test stronger and positive-only steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import html
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_BASELINE_RUN = (
    "current_coef1_pos8_neg8",
    "Current: coefficient 1.0, +8 / -8 experts",
    "experiments/mixtral_steermoe_fears_seed7_question_only_with_llama",
)
DEFAULT_ABLATION_RUNS = [
    (
        "coef4_pos8_neg8",
        "Stronger only: coefficient 4.0, +8 / -8 experts",
        "experiments/mixtral_steermoe_fears_seed7_question_only_coef4_pos8_neg8",
    ),
    (
        "coef1_pos8_neg0",
        "Positive-only only: coefficient 1.0, +8 / -0 experts",
        "experiments/mixtral_steermoe_fears_seed7_question_only_coef1_pos8_neg0",
    ),
    (
        "coef4_pos8_neg0",
        "Combined: coefficient 4.0, +8 / -0 experts",
        "experiments/mixtral_steermoe_fears_seed7_question_only_coef4_pos8_neg0",
    ),
    (
        "coef1_pos10_neg100",
        "Paper faithful-style: coefficient 1.0, +10 / -100 experts",
        "experiments/mixtral_steermoe_fears_seed7_question_only_coef1_pos10_neg100",
    ),
    (
        "coef1_pos20_neg0",
        "Paper safety-style: coefficient 1.0, +20 / -0 experts",
        "experiments/mixtral_steermoe_fears_seed7_question_only_coef1_pos20_neg0",
    ),
    (
        "paper_question_only_eps001_pos20_neg0",
        "Paper rule question-only: epsilon 0.01, +20 / -0 experts",
        "experiments/mixtral_steermoe_fears_seed7_question_only_paper_eps001_pos20_neg0",
    ),
    (
        "paper_prefix_eps001_pos20_neg0",
        "Paper rule visible-prefix sanity: epsilon 0.01, +20 / -0 experts",
        "experiments/mixtral_steermoe_fears_seed7_prefix_conditioned_paper_eps001_pos20_neg0",
    ),
]


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    path: Path


@dataclass
class LoadedRun:
    spec: RunSpec
    plan: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_run(spec: RunSpec) -> LoadedRun:
    plan_path = spec.path / "manual_review_plan.json"
    metadata_path = spec.path / "generation_metadata.json"
    plan = _read_json(plan_path) if plan_path.exists() else None
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    return LoadedRun(spec=spec, plan=plan, metadata=metadata)


def _case_key(case: Dict[str, Any]) -> Tuple[str, int]:
    return str(case["concept"]), int(case["evaluation_version"])


def _case_index(plan: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {_case_key(case): case for case in plan["cases"]}


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _display_response(value: str) -> str:
    return _escape(value.strip())


def _metadata_cell(metadata: Optional[Dict[str, Any]], key: str) -> str:
    if metadata is None:
        return "missing"
    value = metadata.get(key, "")
    return _escape(value)


def _setting_rows(runs: Iterable[LoadedRun]) -> str:
    rows = []
    for run in runs:
        metadata = run.metadata
        rows.append(
            "<tr>"
            f"<td><strong>{_escape(run.spec.label)}</strong></td>"
            f"<td>{_metadata_cell(metadata, 'steering_coefficient')}</td>"
            f"<td>{_metadata_cell(metadata, 'steering_rule')}</td>"
            f"<td>{_metadata_cell(metadata, 'top_positive_experts')}</td>"
            f"<td>{_metadata_cell(metadata, 'top_negative_experts')}</td>"
            f"<td>{_metadata_cell(metadata, 'prompt_mode')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _artifact_rows(runs: Iterable[LoadedRun]) -> str:
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{_escape(run.spec.label)}</td>"
            f"<td><code>{_escape(run.spec.path)}</code></td>"
            "</tr>"
        )
    return "\n".join(rows)


def _run_status_rows(runs: Iterable[LoadedRun]) -> str:
    rows = []
    for run in runs:
        has_plan = run.plan is not None
        has_metadata = run.metadata is not None
        rows.append(
            "<tr>"
            f"<td>{_escape(run.spec.label)}</td>"
            f"<td>{'yes' if has_plan else 'no'}</td>"
            f"<td>{'yes' if has_metadata else 'no'}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _response_rows_for_case(
    base_case: Dict[str, Any],
    ablation_cases: List[Tuple[LoadedRun, Optional[Dict[str, Any]]]],
) -> str:
    base_responses = base_case.get("responses", {})
    rows = [
        ("Llama 3.1 8B baseline", base_responses.get("reference_model", "")),
        ("Mixtral baseline", base_responses.get("mixtral_baseline", "")),
        ("Current SteerMoE: coefficient 1.0, +8 / -8 experts", base_responses.get("mixtral_steermoe", "")),
    ]
    for run, case in ablation_cases:
        response = ""
        if case:
            response = case.get("responses", {}).get("mixtral_steermoe", "")
        rows.append((run.spec.label, response))

    return "\n".join(
        "<tr>"
        f"<th scope='row'>{_escape(label)}</th>"
        f"<td><pre>{_display_response(response)}</pre></td>"
        "</tr>"
        for label, response in rows
    )


def build_report_html(
    baseline_run: LoadedRun,
    ablation_runs: List[LoadedRun],
    title: str,
) -> str:
    all_runs = [baseline_run, *ablation_runs]
    complete_runs = [run for run in all_runs if run.plan is not None]
    if baseline_run.plan is None:
        raise ValueError(f"Baseline run is missing manual_review_plan.json: {baseline_run.spec.path}")

    base_plan = baseline_run.plan
    ablation_indexes = [
        (run, _case_index(run.plan) if run.plan is not None else {})
        for run in ablation_runs
    ]

    case_rows = []
    for case in base_plan["cases"]:
        key = _case_key(case)
        ablation_cases = [(run, index.get(key)) for run, index in ablation_indexes]
        case_rows.append(
            "<section class='case'>"
            f"<h2>{_escape(case['concept'])} - Eval v{_escape(case['evaluation_version'])}</h2>"
            f"<p class='prompt'><strong>Question-only prompt:</strong> {_escape(case['full_prompt_text'])}</p>"
            f"<p class='diagnostic'><strong>Prefix diagnostic:</strong> {_escape(case.get('prefix_conditioned_prompt_text', ''))}</p>"
            "<table class='responses'>"
            "<thead><tr>"
            "<th>Condition</th><th>Response</th>"
            + "</tr></thead>"
            "<tbody>"
            + _response_rows_for_case(case, ablation_cases)
            + "</tbody></table>"
            "</section>"
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5b6570;
      --line: #d8dee6;
      --fill: #f6f8fb;
      --accent: #254f8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 28px;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      line-height: 1.45;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 8px; font-size: 17px; color: var(--accent); }}
    p {{ margin: 6px 0; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .lede {{
      max-width: 960px;
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 18px;
    }}
    .note {{
      background: #fff8df;
      border: 1px solid #ead99a;
      padding: 10px 12px;
      margin: 14px 0 18px;
    }}
    .legend {{
      background: #f6f8fb;
      border: 1px solid var(--line);
      padding: 12px 14px;
      margin: 14px 0 18px;
    }}
    .legend h2 {{
      margin-top: 0;
    }}
    .legend dl {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 6px 12px;
      margin: 8px 0 0;
    }}
    .legend dt {{
      font-weight: 700;
      color: var(--accent);
    }}
    .legend dd {{
      margin: 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
      margin: 10px 0 20px;
    }}
    th, td {{
      border: 1px solid var(--line);
      vertical-align: top;
      padding: 8px;
    }}
    th {{
      background: var(--fill);
      text-align: left;
      font-weight: 650;
    }}
    .settings th, .settings td {{ font-size: 12px; }}
    .settings th:first-child {{
      width: 300px;
    }}
    .artifacts th:first-child {{
      width: 260px;
    }}
    .case {{
      break-inside: avoid;
      page-break-inside: avoid;
      border-top: 2px solid var(--line);
      padding-top: 14px;
      margin-top: 18px;
    }}
    .prompt, .diagnostic {{ color: var(--muted); }}
    .diagnostic {{ font-size: 12px; }}
    .responses {{
      table-layout: auto;
    }}
    .responses th:first-child {{
      width: 230px;
    }}
    .responses th {{ font-size: 12px; }}
    .responses tbody th {{
      background: #fbfcfe;
      color: var(--accent);
      font-weight: 650;
    }}
    .responses pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.4;
    }}
    @media print {{
      body {{ margin: 16mm; }}
      .case {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <h1>{_escape(title)}</h1>
  <p class="lede">
    Question-only Mixtral SteerMoE ablation report. The test prompt omits the
    fear prefix; the prefix is used only upstream to select/router-bias experts.
    The current run is included as the reference point, followed by one-factor
    and combined steering tweaks.
  </p>
  <div class="note">
    Current reference settings:
    <code>STEERING_COEFFICIENT=1.0</code>,
    <code>TOP_POSITIVE_EXPERTS=8</code>,
    <code>TOP_NEGATIVE_EXPERTS=8</code>.
  </div>

  <section class="legend">
    <h2>How To Read This Report</h2>
    <p>
      Each review case asks the same fear-related question. The rows compare
      unsteered baselines against different Mixtral SteerMoE interventions.
      The main question is whether a hidden fear concept appears in the answer
      when the visible prompt is only the bare question.
    </p>
    <dl>
      <dt>Coeff.</dt>
      <dd>
        Steering strength. For <code>additive_bias</code>, this is the router-logit
        amount added to selected experts. For <code>paper</code>, this is the
        paper's epsilon margin.
      </dd>
      <dt>Rule</dt>
      <dd>
        <code>additive_bias</code> is the original local implementation.
        <code>paper</code> applies the SteerMoE paper rule:
        activate experts at row-wise <code>s_max + epsilon</code> and deactivate
        experts at <code>s_min - epsilon</code> after log-softmax.
      </dd>
      <dt>Top +</dt>
      <dd>Number of globally strongest positive-risk experts promoted.</dd>
      <dt>Top -</dt>
      <dd>Number of globally strongest negative-risk experts suppressed.</dd>
      <dt>Prompt Mode</dt>
      <dd>
        <code>question_only</code> hides the fear concept from the visible prompt.
        <code>prefix_conditioned</code> shows the concept in the prompt and is a
        sanity check that the model can express the concept when it sees it.
      </dd>
    </dl>
  </section>

  <h2>Run Status</h2>
  <table class="settings">
    <thead><tr><th>Run</th><th>Plan JSON</th><th>Metadata JSON</th></tr></thead>
    <tbody>{_run_status_rows(all_runs)}</tbody>
  </table>

  <h2>Settings</h2>
  <table class="settings">
    <thead>
      <tr>
        <th>Run</th>
        <th>Coeff.</th>
        <th>Rule</th>
        <th>Top +</th>
        <th>Top -</th>
        <th>Prompt Mode</th>
      </tr>
    </thead>
    <tbody>{_setting_rows(all_runs)}</tbody>
  </table>

  <h2>Review Cases</h2>
  {''.join(case_rows)}

  <h2>Artifact Paths</h2>
  <p class="diagnostic">
    These paths are included for reproducibility. They are not needed to read the
    qualitative comparison above.
  </p>
  <table class="artifacts">
    <thead><tr><th>Run</th><th>Local artifact folder</th></tr></thead>
    <tbody>{_artifact_rows(all_runs)}</tbody>
  </table>
</body>
</html>
"""


def parse_run_spec(value: str) -> RunSpec:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Run specs must be KEY=LABEL=PATH.")
    key, label, path = parts
    return RunSpec(key=key, label=label, path=Path(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Mixtral question-only steering ablation report HTML."
    )
    parser.add_argument(
        "--baseline-run",
        type=parse_run_spec,
        default=RunSpec(
            key=DEFAULT_BASELINE_RUN[0],
            label=DEFAULT_BASELINE_RUN[1],
            path=Path(DEFAULT_BASELINE_RUN[2]),
        ),
        help="Baseline run as KEY=LABEL=PATH.",
    )
    parser.add_argument(
        "--ablation-run",
        type=parse_run_spec,
        action="append",
        default=[
            RunSpec(key=key, label=label, path=Path(path))
            for key, label, path in DEFAULT_ABLATION_RUNS
        ],
        help="Ablation run as KEY=LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--output-html",
        default="experiments/mixtral_steermoe_ablation_summary/ablation_report.html",
        help="Path where the compiled HTML report should be written.",
    )
    parser.add_argument(
        "--title",
        default="Mixtral Question-Only SteerMoE Ablation Report",
        help="Report title.",
    )
    args = parser.parse_args()

    baseline_run = _load_run(args.baseline_run)
    ablation_runs = [_load_run(spec) for spec in args.ablation_run]
    html_text = build_report_html(
        baseline_run=baseline_run,
        ablation_runs=ablation_runs,
        title=args.title,
    )
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text)
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
