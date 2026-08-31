# ageneval-task-deepsearchqa

[DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) — 900
multi-step information-seeking tasks from Google DeepMind. Agents are expected
to search the open web (`web_search` / `open_url`) and return an exhaustive
answer.

- HF: `google/deepsearchqa`, split `eval`
- Official outcome metric in this adapter: `deepsearch_grader` (alias
  `deepsearch_match`) — single-answer containment; set-answer item recall
  (paper Appendix A). Implementation:
  `src/ageneval/task/datasets/deepsearchqa/grader.py`. The paper's autorater
  is `gemini-2.5-flash`; that judge is not wired here.
- `answer_type` is **not** shown to the agent (dataset card requirement).
