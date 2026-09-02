from ageneval.task.datasets.swe_bench_pro.grader import (
    GRADER,
    grade_swe_bench_pro_output,
    score_swe_bench_pro,
)


def test_inline_spec_keeps_live_scale_scorer() -> None:
    assert GRADER.id == "swe_resolved"
    assert GRADER.mode == "inline"
    assert GRADER.grade is score_swe_bench_pro
    assert GRADER.official is True


def test_post_platform_grade_accepts_historical_prefixed_counts() -> None:
    report = grade_swe_bench_pro_output(
        {
            "resolved": False,
            "swe_status": "unresolved",
            "swe_f2p_passed": 1,
            "swe_f2p_total": 2,
            "swe_p2p_passed": 4,
            "swe_p2p_total": 4,
        }
    )

    assert report.score == 0.0
    assert report.metrics == {
        "swe_resolved": 0.0,
        "swe_fail_to_pass": 0.5,
        "swe_pass_to_pass": 1.0,
    }
