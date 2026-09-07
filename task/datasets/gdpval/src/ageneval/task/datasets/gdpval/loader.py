"""GDPval loader — HuggingFace ``openai/gdpval``.

Each row is a deliverable-generation task with a natural-language prompt and
optional reference files. Official GDPval agents receive those files in a
sandbox workspace and read them with tools. This loader:

* resolves every ``reference_files`` entry onto disk (local cache first);
* copies every file into an isolated workspace (E2B when configured);
* lists filenames only — never dumps 12k excerpts or "binary attachment";
* never tells the model that attachments are invisible.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

from ageneval.task.core.dataset import Dataset, TaskInput

logger = logging.getLogger(__name__)

_HF_ID = "openai/gdpval"
_MAX_RUBRIC_CHARS = 6000

_DEFAULT_FILE_ROOTS = (
    Path("/data/agenteval/a2e-data-full-20260817/gdpval-files"),
    Path.home() / ".cache" / "a2e" / "gdpval-files",
    Path("/mnt/shared-storage-user/zhangmingxuan/ageneval/glm53-merge/gdpval-files"),
)


@dataclass
class GDPvalDataset(Dataset):
    name: str
    tasks: Sequence[TaskInput]

    def __iter__(self) -> Iterator[TaskInput]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


def _file_names(raw: object) -> list[str]:
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            s = str(item)
            out.append(os.path.basename(s.rstrip("/")) or s)
    return out


def _attach_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("A2E_GDPVAL_FILES_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(_DEFAULT_FILE_ROOTS)
    seen: set[str] = set()
    uniq: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


@lru_cache(maxsize=8)
def _hashed_index(root: str) -> dict[str, str]:
    idx: dict[str, str] = {}
    hashed = Path(root) / "reference_files"
    if not hashed.is_dir():
        return idx
    for path in hashed.rglob("*"):
        if path.is_file():
            idx.setdefault(path.name, str(path))
    return idx


def _fetch_reference_file(rel_path: str) -> Path | None:
    """Resolve a GDPval reference file from local caches, then the Hub."""
    rel = str(rel_path).lstrip("/")
    name = os.path.basename(rel)
    for root in _attach_roots():
        for cand in (
            root / rel,
            root / name,
            root / "reference_files" / rel,
            root / "reference_files" / name,
        ):
            if cand.is_file():
                return cand
        hit = _hashed_index(str(root)).get(name)
        if hit:
            return Path(hit)
    if os.environ.get("A2E_GDPVAL_FILES", "1") == "0":
        return None
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        dest = _attach_roots()[0]
        dest.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_HF_ID,
            repo_type="dataset",
            filename=rel,
            local_dir=str(dest),
        )
        fetched = Path(path)
        return fetched if fetched.is_file() else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDPval attachment %s unavailable (%s)", rel, str(exc)[:160])
        return None


def _build_instruction(prompt: str, files: dict[str, str], missing: list[str]) -> str:
    instruction = prompt.strip()
    if files:
        listed = "\n".join(f"  - {name} (in sandbox; call read_file / view_image / code_exec)" for name in files)
        instruction += (
            "\n\n[Reference files] These files are in the sandbox workspace. "
            "Read them with tools. Do not invent file contents.\n"
            f"{listed}"
        )
    if missing:
        listed = "\n".join(f"  - {name}" for name in missing)
        instruction += (
            "\n\n[Unresolved reference file names] Could not copy these onto the "
            "sandbox disk:\n"
            f"{listed}"
        )
    return instruction


def _local_gdpval_parquets() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("A2E_GDPVAL_PARQUET")
    if env:
        paths.append(Path(env))
    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub" / "datasets--openai--gdpval"
    if hub.is_dir():
        paths.extend(sorted(hub.glob("snapshots/*/data/train-*.parquet")))
    return paths


def _load_gdpval_split(split: str):
    for path in _local_gdpval_parquets():
        resolved = path.resolve() if path.exists() else None
        if resolved is None or not resolved.is_file():
            continue
        import pyarrow.parquet as pq

        logger.info("GDPval loader: local parquet %s", resolved)
        return pq.read_table(resolved).to_pylist()
    from datasets import load_dataset  # Hub fallback only

    return load_dataset(_HF_ID, split=split, streaming=False)


def load_gdpval_tasks(split: str = "train", n: int | None = 5) -> GDPvalDataset:
    ds = _load_gdpval_split(split)
    tasks: list[TaskInput] = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        prompt = str(row.get("prompt", "") or "")
        rubric = str(row.get("rubric_pretty", "") or "")[:_MAX_RUBRIC_CHARS]
        ref_rels = [str(p) for p in (row.get("reference_files") or [])]
        ref_names = _file_names(ref_rels)
        found_src: dict[str, str] = {}
        missing: list[str] = []
        for rel, name in zip(ref_rels, ref_names):
            fetched = _fetch_reference_file(rel)
            if fetched is None:
                missing.append(name)
                continue
            found_src[name] = str(fetched)
        from ageneval.task.datasets.gdpval.sandbox import open_gdp_sandbox

        box = open_gdp_sandbox()
        found: dict[str, str] = {}
        for name, src in found_src.items():
            dest = box.put_file(name, src)
            local = box.workspace / name
            found[name] = str(local if local.is_file() else dest)
        task_id = str(row.get("task_id") or f"gdpval-{i:04d}")
        tasks.append(
            TaskInput(
                task_id=task_id,
                instruction=_build_instruction(prompt, found, missing),
                initial_state={
                    "reference_files": found,
                    "reference_names": ref_names,
                    "missing_reference_files": missing,
                    "workspace": str(box.workspace),
                    "sandbox_backend": box.backend,
                    "_gdp_sandbox": box,
                },
                expected_outputs=(rubric,) if rubric else (),
                metadata={
                    "dataset": "gdpval-aa",
                    "sector": str(row.get("sector", "")),
                    "occupation": str(row.get("occupation", "")),
                    "n_reference_files": len(ref_names),
                    "n_attachments_loaded": len(found),
                },
            )
        )
    logger.info("GDPval loader: %s (%s), %s tasks", _HF_ID, split, len(tasks))
    return GDPvalDataset(name=f"gdpval-{split}", tasks=tasks)
