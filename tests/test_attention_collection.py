import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import (
    compute_inserted_token_span_from_ids,
    compute_prefix_attention_sums_for_last_n,
    summarize_layer_to_token_index,
)
from moe_attention_guided_steering.upstream_prompt_datasets import (
    build_upstream_statement_prompt_pairs,
    get_upstream_positive_prefix,
)


class AttentionCollectionTestCase(unittest.TestCase):
    """Verify the upstream-style prompt builder and attention summarization logic."""

    def test_compute_inserted_token_span_from_ids_recovers_prefix(self) -> None:
        full_ids = [10, 11, 21, 22, 30, 31, 40, 41]
        reduced_ids = [10, 11, 30, 31, 40, 41]
        self.assertEqual(compute_inserted_token_span_from_ids(full_ids, reduced_ids), (2, 4))

    def test_compute_prefix_attention_sums_for_last_n(self) -> None:
        layer_attention = [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.2, 0.3, 0.3],
                    [0.5, 0.1, 0.2, 0.2],
                ],
            ]
        ]

        scores = compute_prefix_attention_sums_for_last_n(
            layer_attention=layer_attention,
            num_candidate_suffix_tokens=2,
            prefix_start=0,
            prefix_end=2,
            head_aggregation="mean",
        )
        self.assertEqual(scores, [0.55, 0.45])

    def test_summarize_layer_to_token_index_uses_negative_indexing(self) -> None:
        from moe_attention_guided_steering.attention_collection import PromptAttentionTrace

        trace_a = PromptAttentionTrace(
            prompt_id="a",
            concept_type="fears",
            concept_value="Bugs",
            statement_index=0,
            statement_text="x",
            prefix_text="prefix",
            body_text="body",
            full_prompt_text="prefixbody",
            prefix_token_span=(1, 2),
            candidate_token_texts=["tokA", "tokB", "tokC"],
            candidate_token_relative_indices=[-3, -2, -1],
            layer_attention_scores=[[0.1, 0.9, 0.3], [0.8, 0.2, 0.1]],
        )
        trace_b = PromptAttentionTrace(
            prompt_id="b",
            concept_type="fears",
            concept_value="Bugs",
            statement_index=2,
            statement_text="y",
            prefix_text="prefix",
            body_text="body",
            full_prompt_text="prefixbody",
            prefix_token_span=(1, 2),
            candidate_token_texts=["tokA", "tokB", "tokC"],
            candidate_token_relative_indices=[-3, -2, -1],
            layer_attention_scores=[[0.7, 0.4, 0.6], [0.1, 0.3, 0.95]],
        )

        self.assertEqual(
            summarize_layer_to_token_index([trace_a, trace_b]),
            {0: -2, 1: -1},
        )

    def test_build_upstream_statement_prompt_pairs_for_fears(self) -> None:
        pairs = build_upstream_statement_prompt_pairs(
            concept_type="fears",
            concept_value="Bugs",
            general_statements_by_class={
                "class_0": ["Alpha"],
                "class_1": ["Beta"],
            },
            statement_stride=1,
        )

        self.assertEqual(get_upstream_positive_prefix("fears", "Bugs"), "Personify someone who is terrified of Bugs.  ")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(pairs[0].positive_full_prompt.startswith("Personify someone who is terrified of Bugs."))
        self.assertTrue(pairs[0].negative_full_prompt.startswith("What are your thoughts"))


if __name__ == "__main__":
    unittest.main()
