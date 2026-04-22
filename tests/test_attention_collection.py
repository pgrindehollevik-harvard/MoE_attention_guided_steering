import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import (
    HFModelResources,
    _flatten_token_ids,
    collect_attention_trace_for_prompt_pair,
    compute_inserted_token_span_from_ids,
    compute_prefix_attention_sums_for_last_n,
    summarize_layer_to_token_index,
)
from moe_attention_guided_steering.upstream_prompt_datasets import (
    StatementPromptPair,
    build_upstream_statement_prompt_pairs,
    get_upstream_positive_prefix,
)


class AttentionCollectionTestCase(unittest.TestCase):
    """Verify the upstream-style prompt builder and attention summarization logic."""

    def test_flatten_token_ids_accepts_mapping_outputs(self) -> None:
        self.assertEqual(_flatten_token_ids({"input_ids": [[7, 8, 9]]}), [7, 8, 9])

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

    def test_collect_attention_trace_uses_chat_formatted_inputs(self) -> None:
        class DummyEncoding(dict):
            def to(self, device: str) -> "DummyEncoding":
                return self

        class DummyTokenizer:
            def apply_chat_template(
                self,
                chat,
                tokenize: bool,
                add_generation_prompt: bool,
                return_tensors: str,
            ) -> DummyEncoding:
                self.last_chat = chat
                text = chat[0]["content"]
                if text == "prefixbody":
                    return DummyEncoding(
                        {
                            "input_ids": [[10, 20, 21, 30, 31, 99]],
                            "attention_mask": [[1, 1, 1, 1, 1, 1]],
                        }
                    )
                if text == "body":
                    return DummyEncoding(
                        {
                            "input_ids": [[10, 30, 31, 99]],
                            "attention_mask": [[1, 1, 1, 1]],
                        }
                    )
                raise AssertionError(f"Unexpected chat text: {text}")

            def __call__(self, *args, **kwargs):
                raise AssertionError("collect_attention_trace_for_prompt_pair should not bypass the chat template.")

            def convert_ids_to_tokens(self, ids):
                return [f"tok_{token_id}" for token_id in ids]

        class DummyOutputs:
            attentions = [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.4, 0.3, 0.1, 0.1, 0.1],
                            [0.0, 0.05, 0.15, 0.2, 0.2, 0.2],
                        ]
                    ]
                ]
            ]

        class DummyModel:
            device = "cpu"

            def __call__(self, **kwargs):
                self.last_kwargs = kwargs
                return DummyOutputs()

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        dummy_torch = types.SimpleNamespace(no_grad=lambda: _NoGrad())
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = dummy_torch
        try:
            tokenizer = DummyTokenizer()
            model = DummyModel()
            trace = collect_attention_trace_for_prompt_pair(
                prompt_pair=StatementPromptPair(
                    prompt_id="fears:Bugs:0",
                    concept_type="fears",
                    concept_value="Bugs",
                    statement_index=0,
                    statement_text="statement",
                    positive_prefix_text="prefix",
                    body_text="body",
                    positive_full_prompt="prefixbody",
                    negative_full_prompt="body",
                ),
                resources=HFModelResources(
                    model=model,
                    tokenizer=tokenizer,
                    model_id="demo",
                    model_tag="demo",
                    num_candidate_suffix_tokens=2,
                ),
                head_aggregation="mean",
            )
        finally:
            if previous_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = previous_torch

        self.assertEqual(model.last_kwargs["input_ids"], [[10, 20, 21, 30, 31, 99]])
        self.assertEqual(model.last_kwargs["attention_mask"], [[1, 1, 1, 1, 1, 1]])
        self.assertEqual(trace.prefix_token_span, (1, 3))
        self.assertEqual(trace.candidate_token_texts, ["tok_31", "tok_99"])
        self.assertEqual(trace.layer_attention_scores, [[0.7, 0.2]])


if __name__ == "__main__":
    unittest.main()
