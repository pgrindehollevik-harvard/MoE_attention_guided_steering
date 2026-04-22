"""Core package for the MoE attention-guided steering scaffold."""

from .attention_collection import (
    collect_attention_run,
    infer_candidate_suffix_token_count,
    load_hf_model_resources,
    save_attention_run,
)
from .config import ExperimentConfig, InterventionConfig, SelectionConfig
from .manual_review import build_manual_review_plan, extract_evaluation_question
from .pipeline import PipelineArtifacts, run_pipeline
from .reference_data import (
    ReferenceConceptSuite,
    ReferenceDataBundle,
    load_reference_concept_suite,
    load_reference_data,
)
from .upstream_prompt_datasets import build_upstream_statement_prompt_pairs

__all__ = [
    "ExperimentConfig",
    "InterventionConfig",
    "PipelineArtifacts",
    "ReferenceConceptSuite",
    "ReferenceDataBundle",
    "SelectionConfig",
    "build_manual_review_plan",
    "build_upstream_statement_prompt_pairs",
    "collect_attention_run",
    "extract_evaluation_question",
    "infer_candidate_suffix_token_count",
    "load_hf_model_resources",
    "load_reference_concept_suite",
    "load_reference_data",
    "run_pipeline",
    "save_attention_run",
]
