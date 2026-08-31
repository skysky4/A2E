# ageneval-task-tau2

τ2-bench — Sierra Research's next-gen agent benchmark (multi-turn user simulator
+ tool calling, richer than τ-bench). Official outcome metric is ``tau_grader`` (alias ``tau_reward``)
(Sierra ``calculate_reward`` pass^1). No public PyPI release at this moment; the
adapter mirrors τ-bench's structure and falls back to vendor when the upstream
is unavailable.
