"""Core package for the MoE attention-guided steering scaffold."""

from .attention_collection import (
    collect_attention_run,
    infer_candidate_suffix_token_count,
    load_hf_model_resources,
    save_attention_run,
)
from .generation import generate_unsteered_response
from .manual_review import build_manual_review_plan, extract_evaluation_question
from .model_comparison import (
    build_empty_model_comparison_results,
    load_model_comparison_specs,
    render_model_comparison_html,
    render_model_comparison_markdown,
)
from .mixtral_backend import (
    collect_mixtral_paired_routing_traces,
    collect_mixtral_target_routing_trace_for_messages,
    fill_manual_review_plan_with_mixtral_generations,
    generate_with_mixtral_steering,
)
from .olmoe_backend import (
    build_steermoe_activation_table_from_paired_traces,
    build_steermoe_replication_plan,
    collect_olmoe_paired_routing_traces,
    collect_olmoe_target_routing_trace_for_messages,
    fill_manual_review_plan_with_steermoe_generations,
    generate_with_olmoe_steering,
    steermoe_plan_to_router_bias_by_layer,
    steermoe_plan_to_router_steering_by_layer,
)
from .reference_data import (
    ReferenceConceptSuite,
    ReferenceDataBundle,
    load_reference_concept_suite,
    load_reference_data,
)
from .upstream_prompt_datasets import (
    build_concept_conditioned_evaluation_prompt,
    build_custom_steering_examples_from_statement_prompt_pairs,
    build_upstream_statement_prompt_pairs,
)

__all__ = [
    "ReferenceConceptSuite",
    "ReferenceDataBundle",
    "build_manual_review_plan",
    "build_concept_conditioned_evaluation_prompt",
    "build_empty_model_comparison_results",
    "build_custom_steering_examples_from_statement_prompt_pairs",
    "build_steermoe_activation_table_from_paired_traces",
    "build_steermoe_replication_plan",
    "build_upstream_statement_prompt_pairs",
    "collect_attention_run",
    "collect_mixtral_paired_routing_traces",
    "collect_mixtral_target_routing_trace_for_messages",
    "collect_olmoe_paired_routing_traces",
    "collect_olmoe_target_routing_trace_for_messages",
    "extract_evaluation_question",
    "fill_manual_review_plan_with_mixtral_generations",
    "fill_manual_review_plan_with_steermoe_generations",
    "generate_unsteered_response",
    "generate_with_mixtral_steering",
    "generate_with_olmoe_steering",
    "infer_candidate_suffix_token_count",
    "load_hf_model_resources",
    "load_model_comparison_specs",
    "load_reference_concept_suite",
    "load_reference_data",
    "render_model_comparison_html",
    "render_model_comparison_markdown",
    "save_attention_run",
    "steermoe_plan_to_router_bias_by_layer",
    "steermoe_plan_to_router_steering_by_layer",
]
