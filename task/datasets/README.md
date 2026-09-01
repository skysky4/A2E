# Benchmark profiles

Each benchmark owns its execution contract. A dataset package with one
benchmark stores it in `benchmark.yaml`; packages that expose several variants
store one file per variant under `benchmarks/`.

The profile declares the benchmark id and aliases, loader, binding, grader,
optional session/setup hooks, default budgets, agent overrides, and resource
requirements. Entrypoints use `python.module:attribute` syntax and are imported
lazily by `ageneval.task.runners`.

Model connectivity and concurrency remain in `task/models/*.yaml`. A Campaign
selects benchmark ids, model ids, harnesses, samples, and global scheduling
limits. When a new Campaign is prepared, the full benchmark profile and its
digest are copied into `lock.json`; resuming fails if that profile changed.

Campaign-level values are explicit overrides. For example,
`execution.timeout_seconds` overrides the selected benchmark's
`defaults.run_deadline`; when it is omitted, the benchmark value is used.
