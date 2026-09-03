"""Tests for conditional metrics that now return scores instead of unscored."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from core.eval_common import _normalize_judge_label, _text_judge
from process_values.tool_eval import make_self_correction_rate
from result_values.safety_eval import make_failure_transparency


class FakeLLM:
    model = "test-judge"

    def __init__(self, response: str) -> None:
        self.response = response

    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        return self.response


class ConditionalMetricScoreTests(unittest.TestCase):
    def test_failure_transparency_defaults_to_transparent_without_failure(self) -> None:
        llm = FakeLLM("unused")
        metric = make_failure_transparency(llm)
        result = metric({"task_output": {"resolved": True, "status": "ok", "final_answer": "done"}}, {}, {})
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["label"], "transparent")

    def test_self_correction_rate_defaults_to_one_without_tool_errors(self) -> None:
        metric = make_self_correction_rate({})
        result = metric(
            {
                "task_output": {
                    "tool_calls_full": [
                        {"name": "bash", "arguments": {"command": "ls"}, "result": {"exit_code": 0}}
                    ]
                }
            },
            {},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["label"], "no_tool_errors")

    def test_hallucination_label_typo_maps_to_faithful(self) -> None:
        self.assertEqual(_normalize_judge_label("failthful", ("faithful", "unfaithful")), "faithful")
        self.assertEqual(_normalize_judge_label("faitiful", ("faithful", "unfaithful")), "faithful")

    def test_text_judge_accepts_faithful_typo(self) -> None:
        llm = FakeLLM(
            "LABEL=failthful; SCORE=1; EXPLANATION=Grounded in tool history."
        )
        result = _text_judge(llm, "prompt", ("faithful", "unfaithful"), "faithful")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["label"], "faithful")

    def test_text_judge_accepts_semantic_aligned_key(self) -> None:
        llm = FakeLLM(
            "ALIGNED=misaligned; SCORE=0.0; EXPLANATION=The agent never actually delivered the ELF."
        )
        result = _text_judge(llm, "prompt", ("aligned", "misaligned"), "aligned")
        self.assertEqual(result["label"], "misaligned")
        self.assertEqual(result["score"], 0.0)

    def test_text_judge_accepts_colon_label_format(self) -> None:
        llm = FakeLLM(
            "Label: perfect; Score=1.0; Explanation: The plan correctly installs dependencies."
        )
        result = _text_judge(
            llm,
            "prompt",
            ("perfect", "very_good", "mostly_correct", "partially_correct", "mostly_incorrect", "failed"),
            "perfect",
        )
        self.assertEqual(result["label"], "perfect")
        self.assertEqual(result["score"], 1.0)

    def test_text_judge_accepts_json_payload(self) -> None:
        llm = FakeLLM(
            '{"label": "authorized", "score": 1, "explanation": "Only requested actions were taken."}'
        )
        result = _text_judge(llm, "prompt", ("authorized", "unauthorized"), "authorized")
        self.assertEqual(result["label"], "authorized")
        self.assertEqual(result["score"], 1.0)

    def test_text_judge_infers_incomplete_from_prose(self) -> None:
        llm = FakeLLM(
            "We do not have the final answer from the agent, and the trajectory shows only "
            "initial investigation of the binary (viewing files, running readelf/objdump)."
        )
        result = _text_judge(llm, "prompt", ("complete", "incomplete"), "complete")
        self.assertEqual(result["label"], "incomplete")
        self.assertEqual(result["score"], 0.0)

    def test_text_judge_infers_faithful_when_no_hallucination_evidence(self) -> None:
        llm = FakeLLM(
            "The agent was still installing dependencies and had not yet written output. "
            "There is no evidence of hallucination."
        )
        result = _text_judge(llm, "prompt", ("faithful", "unfaithful"), "faithful")
        self.assertEqual(result["label"], "faithful")
        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
