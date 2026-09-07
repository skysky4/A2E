# ageneval-task-gdpval

GDPval dataset adapter for A2E.

**Source:** [`openai/gdpval`](https://huggingface.co/datasets/openai/gdpval) (OpenAI).

Official GDPval agents use a computer-use sandbox (E2B / production Docker)
with:

* reference files on disk (never dumped into the prompt)
* Web Search / Web Fetch
* View Image
* Code Exec
* Finish / Abandon
* up to 250 turns
* submit one or more **real files**

This adapter starts E2B when `E2B_API_KEY` is set; otherwise it uses an
isolated per-task workspace and still copies every attachment onto disk.
Pairwise Elo (human = 1000) remains off-run; the in-run grader is `gdp_grader`.

Resolve files from `A2E_GDPVAL_FILES_DIR` or
`/data/agenteval/a2e-data-full-20260817/gdpval-files`.
