from ageneval.task.datasets.swe_bench.grader import (
    GRADER,
    grade_swe_bench_output,
    score_swe_bench,
)


def test_inline_spec_keeps_live_scorer() -> None:
    assert GRADER.id == "swe_resolved"
    assert GRADER.mode == "inline"
    assert GRADER.grade is score_swe_bench
    assert GRADER.official is True


def test_post_platform_grade_uses_existing_counts() -> None:
    report = grade_swe_bench_output(
        {
            "resolved": True,
            "status": "FULL",
            "f2p_passed": 2,
            "f2p_total": 2,
            "p2p_passed": 3,
            "p2p_total": 4,
        }
    )

    assert report.score == 1.0
    assert report.metrics == {
        "swe_resolved": 1.0,
        "swe_fail_to_pass": 1.0,
        "swe_pass_to_pass": 0.75,
    }
