from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .attention_collection import load_hf_model_resources
from .generation import generate_unsteered_response
from .manual_review import ManualReviewPlan
from .upstream_prompt_datasets import build_concept_conditioned_evaluation_prompt


@dataclass
class ModelComparisonSpec:
    """One model entry for the prefix-conditioned suitability comparison."""

    label: str
    model_id: str
    model_tag: str
    prompt_format: str = "chat"
    plain_template: str = "{prompt}"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = "eager"
    device_map: Optional[str] = "auto"
    post_load_device: Optional[str] = None
    disable_torch_distribution_validation: bool = False
    use_cache: bool = True


def model_comparison_spec_from_dict(data: Dict[str, Any]) -> ModelComparisonSpec:
    """Normalize one JSON model specification into a typed config."""
    return ModelComparisonSpec(
        label=str(data["label"]),
        model_id=str(data["model_id"]),
        model_tag=str(data["model_tag"]),
        prompt_format=str(data.get("prompt_format", "chat")),
        plain_template=str(data.get("plain_template", "{prompt}")),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
        attn_implementation=data.get("attn_implementation", "eager"),
        device_map=data.get("device_map", "auto"),
        post_load_device=data.get("post_load_device"),
        disable_torch_distribution_validation=bool(
            data.get("disable_torch_distribution_validation", False)
        ),
        use_cache=bool(data.get("use_cache", True)),
    )


def load_model_comparison_specs(path: str) -> List[ModelComparisonSpec]:
    """Load model comparison specs from a small JSON config file."""
    data = json.loads(Path(path).read_text())
    return [model_comparison_spec_from_dict(item) for item in data["models"]]


def prompt_text_for_model_comparison_case(
    plan: ManualReviewPlan,
    concept: str,
    evaluation_question: str,
    prompt_mode: str = "prefix",
) -> str:
    """Build the exact model-facing prompt for one comparison case."""
    if prompt_mode == "prefix":
        return build_concept_conditioned_evaluation_prompt(
            concept_type=plan.concept_type,
            concept_value=concept,
            evaluation_question=evaluation_question,
        )
    if prompt_mode == "question":
        return evaluation_question
    raise ValueError("prompt_mode must be either 'prefix' or 'question'.")


def build_empty_model_comparison_results(
    plan: ManualReviewPlan,
    model_specs: Sequence[ModelComparisonSpec],
    prompt_mode: str = "prefix",
) -> Dict[str, Any]:
    """Create the serializable result skeleton shared by dry tests and runs."""
    return {
        "prompt_mode": prompt_mode,
        "prompt_contract": (
            "prefix + evaluation question"
            if prompt_mode == "prefix"
            else "evaluation question only"
        ),
        "concept_type": plan.concept_type,
        "evaluation_family": plan.evaluation_family,
        "seed": plan.seed,
        "models": [asdict(spec) for spec in model_specs],
        "cases": [
            {
                "concept": case.concept,
                "evaluation_version": case.evaluation_version,
                "evaluation_question": case.evaluation_question,
                "prompt_text": prompt_text_for_model_comparison_case(
                    plan=plan,
                    concept=case.concept,
                    evaluation_question=case.evaluation_question,
                    prompt_mode=prompt_mode,
                ),
                "responses": {spec.model_tag: "" for spec in model_specs},
            }
            for case in plan.cases
        ],
    }


def fill_model_comparison_results_with_generations(
    results: Dict[str, Any],
    model_specs: Sequence[ModelComparisonSpec],
    cache_dir: Optional[str] = None,
    device_map_override: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    attn_implementation_override: Optional[str] = None,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    """Generate unsteered responses for each model, one model loaded at a time."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    for spec in model_specs:
        resources = load_hf_model_resources(
            model_id=spec.model_id,
            model_tag=spec.model_tag,
            cache_dir=cache_dir,
            device_map=(
                device_map_override
                if device_map_override is not None
                else spec.device_map
            ),
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            attn_implementation=(
                attn_implementation_override
                if attn_implementation_override is not None
                else spec.attn_implementation
            ),
            trust_remote_code=spec.trust_remote_code,
            infer_attention_suffix_tokens=False,
            post_load_device=spec.post_load_device,
            disable_torch_distribution_validation=spec.disable_torch_distribution_validation,
        )
        try:
            for case in tqdm(results["cases"], desc=f"Generating {spec.model_tag}"):
                case["responses"][spec.model_tag] = generate_unsteered_response(
                    prompt_text=case["prompt_text"],
                    resources=resources,
                    prompt_format=spec.prompt_format,
                    plain_template=spec.plain_template,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=spec.use_cache,
                )
        finally:
            del resources
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - torch is present on GPU runs
                pass

    return results


def render_model_comparison_markdown(results: Dict[str, Any]) -> str:
    """Render a lightweight Markdown report for model-suitability review."""
    model_labels = {model["model_tag"]: model["label"] for model in results["models"]}
    lines = [
        "# Prefix-Conditioned Model Comparison",
        "",
        f"- Prompt mode: **{results['prompt_contract']}**",
        f"- Concept family: **{results['concept_type']}**",
        f"- Evaluation family: **{results['evaluation_family']}**",
        f"- Sampling seed: **{results['seed']}**",
        "",
        "This diagnostic asks whether each base model can follow the explicit concept prefix. It is separate from the steering test, where the prefix is omitted.",
        "",
    ]

    for case in results["cases"]:
        lines.extend(
            [
                f"## {case['concept']} - Eval v{case['evaluation_version']}",
                "",
                f"- Prompt: {case['prompt_text']}",
                "",
            ]
        )
        for model_tag, response in case["responses"].items():
            lines.extend(
                [
                    f"### {model_labels.get(model_tag, model_tag)}",
                    "",
                    response.strip() or "Pending generation",
                    "",
                ]
            )

    return "\n".join(lines)


def render_model_comparison_html(results: Dict[str, Any]) -> str:
    """Render a browsable HTML report for prefix-conditioned model comparison."""
    model_labels = {model["model_tag"]: model["label"] for model in results["models"]}
    sections: List[str] = []
    for case in results["cases"]:
        response_blocks = []
        for model_tag, response in case["responses"].items():
            display = html.escape(response) if response.strip() else "<span class='pending'>Pending generation</span>"
            response_blocks.append(
                f"""
                <div class='response'>
                  <p class='lbl'>{html.escape(model_labels.get(model_tag, model_tag))}</p>
                  <pre>{display}</pre>
                </div>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(case['concept'])} &middot; Eval v{case['evaluation_version']}</h2>
              <p><strong>Prompt:</strong> <code>{html.escape(case['prompt_text'])}</code></p>
              <div class='responses'>{''.join(response_blocks)}</div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prefix-Conditioned Model Comparison</title>
  <style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;line-height:1.5;color:#24292f;padding:0 1rem;}}
h1{{font-size:1.45rem;}} h2{{font-size:1.05rem;margin:0 0 .5rem;}}
section{{border:1px solid #d0d7de;border-radius:8px;padding:1rem;margin-bottom:1.5rem;background:#fafafa;}}
.responses{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;}}
.response{{background:#fff;border-left:4px solid #0550ae;border-radius:8px;padding:.75rem;}}
.lbl{{font-weight:700;color:#0550ae;margin:.1rem 0 .4rem;}}
pre{{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;padding:.6rem;border-radius:6px;font-size:.85rem;margin:.3rem 0 .1rem;min-height:4.5rem;}}
code{{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;}}
.note{{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:.6rem 1rem;margin:1rem 0;font-size:.9rem;}}
.pending{{color:#8a5a00;font-style:italic;}}
  </style>
</head>
<body>
<h1>Prefix-Conditioned Model Comparison</h1>
<div class='note'>
This diagnostic compares base model behavior when the concept prefix is explicitly included.
It does not test steering; the steering review omits the prefix and uses router intervention instead.
</div>
<p>Prompt mode: <strong>{html.escape(results['prompt_contract'])}</strong> &middot; Concept family: <code>{html.escape(results['concept_type'])}</code> &middot; Evaluation family: <code>{html.escape(results['evaluation_family'])}</code> &middot; Seed: <code>{results['seed']}</code></p>
{''.join(sections)}
</body>
</html>
"""
