from ageneval.task.datasets.terminal_bench_2.grader import (
    GRADER,
    grade_terminal_bench_2_output,
    score_terminal_bench_2,
)


def test_inline_spec_and_post_platform_metrics() -> None:
    report = grade_terminal_bench_2_output(
        {
            "resolved": True,
            "tb_reward": "1",
            "status": "graded",
            "tb_tests_total": 3,
            "tb_tests_passed": 3,
            "tb_tests_failed": 0,
        }
    )

    assert GRADER.id == "tb_resolved"
    assert GRADER.mode == "inline"
    assert GRADER.grade is score_terminal_bench_2
    assert GRADER.official is True
    assert report.score == 1.0
    assert report.metrics == {"tb_resolved": 1.0, "tb_reward": 1.0}
