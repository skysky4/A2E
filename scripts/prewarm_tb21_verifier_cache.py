#!/usr/bin/env python3
"""Populate the Docker volumes used by Terminal-Bench 2.1 verifiers."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "task/datasets/terminal_bench_2_1/src/ageneval/task/datasets/terminal_bench_2_1/vendor/tasks"
IMAGE = "ghcr.io/astral-sh/uv:0.9.5-debian"
UV_VERSION = "0.9.5"
DEFAULT_BIN_DIR = ROOT / ".a2e-cache" / "tb21-verifier" / f"uv-{UV_VERSION}"
VOLUMES = (
    "aep-tb21-uv-cache-v1:/root/.cache/uv",
    "aep-tb21-uv-data-v1:/root/.local/share/uv",
)
PIP_VOLUME = "aep-tb21-pip-cache-v1:/root/.cache/pip"


def _verify_uv_binaries(directory: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("uv", "uvx"):
        binary = directory / name
        if not binary.is_file():
            raise FileNotFoundError(binary)
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        expected = f"{name} {UV_VERSION}"
        if result.returncode or not result.stdout.strip().startswith(expected):
            raise RuntimeError(
                f"expected {expected}, got {(result.stdout or result.stderr).strip()!r}"
            )
        versions[name] = result.stdout.strip()
    return versions


def prepare_uv_binaries(image: str, target: Path, *, dry_run: bool) -> None:
    """Extract pinned uv tools for score-time injection, never agent mounting."""
    commands = [
        ["docker", "create", "--entrypoint", "/bin/true", image],
        ["docker", "cp", "<container>:/usr/local/bin/uv", f"{target}/uv"],
        ["docker", "cp", "<container>:/usr/local/bin/uvx", f"{target}/uvx"],
    ]
    print(f"Preparing trusted uv binaries in {target}", flush=True)
    if dry_run:
        for command in commands:
            print(shlex.join(command))
        return

    if (target / "uv").is_file() and (target / "uvx").is_file():
        _verify_uv_binaries(target)
        print("Trusted uv 0.9.5 binaries already exist; keeping them.", flush=True)
        return
    if target.exists():
        raise RuntimeError(f"refusing incomplete trusted uv directory: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    container_id = ""
    with tempfile.TemporaryDirectory(prefix="uv-stage-", dir=target.parent) as temporary:
        staging = Path(temporary)
        created = subprocess.run(
            commands[0], capture_output=True, text=True, check=False
        )
        if created.returncode or not created.stdout.strip():
            raise RuntimeError(f"docker create failed: {created.stderr[-1000:]}")
        container_id = created.stdout.strip()
        try:
            for name in ("uv", "uvx"):
                errors: list[str] = []
                for source in (f"/usr/local/bin/{name}", f"/{name}"):
                    copied = subprocess.run(
                        ["docker", "cp", f"{container_id}:{source}", str(staging / name)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if copied.returncode == 0:
                        break
                    errors.append(copied.stderr[-500:])
                else:
                    raise RuntimeError(
                        f"docker cp {name} failed from known image paths: {errors}"
                    )
                (staging / name).chmod(0o555)

            versions = _verify_uv_binaries(staging)
            (staging / "manifest.json").write_text(
                json.dumps({"image": image, "versions": versions}, indent=2) + "\n",
                encoding="utf-8",
            )
            Path(temporary).replace(target)
            print(f"Extracted and verified uv/uvx {UV_VERSION}.", flush=True)
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def uvx_commands(task_ids: set[str] | None = None) -> list[tuple[str, list[str]]]:
    """Return one harmless invocation for every distinct verifier dependency set."""
    found: dict[tuple[str, ...], str] = {}
    pattern = re.compile(r"(?ms)^uvx\s+\\\n(.*?)(?=\n\s*\n|\nif\s)")
    for test_sh in sorted(TASKS.glob("*/tests/test.sh")):
        task_name = test_sh.parent.parent.name
        if task_ids is not None and task_name not in task_ids:
            continue
        text = test_sh.read_text(encoding="utf-8")
        match = pattern.search(text)
        if not match:
            continue
        command = "uvx " + match.group(1).replace("\\\n", " ")
        command = command.replace("${COMMIT_HASH}", "34bbbfdface3c18e5221aa7de6032d7220c6c6a1")
        tokens = shlex.split(command)
        try:
            pytest_at = tokens.index("pytest", 1)
        except ValueError as exc:
            raise RuntimeError(f"cannot locate pytest executable in {test_sh}") from exc
        warm = tuple([*tokens[: pytest_at + 1], "--version"])
        found.setdefault(warm, task_name)

    # mailman uses uv venv/uv pip instead of uvx, but the same package cache can
    # be populated with an equivalent ephemeral uvx environment.
    mailman = (
        "uvx", "-p", "3.12", "-w", "pytest==8.4.1", "-w", "mailman==3.3.8",
        "-w", "pytest-json-ctrf==0.3.5", "pytest", "--version",
    )
    if task_ids is None or "mailman" in task_ids:
        found.setdefault(mailman, "mailman")
    return sorted(((task, list(cmd)) for cmd, task in found.items()), key=lambda item: item[0])


def pip_commands(task_ids: set[str] | None = None) -> list[tuple[str, str, list[str]]]:
    """Return task image and install arguments for direct-pip verifiers."""
    commands: list[tuple[str, str, list[str]]] = []
    for test_sh in sorted(TASKS.glob("*/tests/test.sh")):
        task_name = test_sh.parent.parent.name
        if task_ids is not None and task_name not in task_ids:
            continue
        text = test_sh.read_text(encoding="utf-8")
        match = re.search(r"(?m)^pip install (.+)$", text)
        if not match:
            continue
        task_dir = test_sh.parent.parent
        task_toml = (task_dir / "task.toml").read_text(encoding="utf-8")
        image_match = re.search(r'(?m)^docker_image\s*=\s*"([^"]+)"', task_toml)
        if not image_match:
            raise RuntimeError(f"cannot locate docker_image in {task_dir / 'task.toml'}")
        commands.append((task_dir.name, image_match.group(1), shlex.split(match.group(1))))
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument(
        "--binaries-only",
        action="store_true",
        help="extract trusted uv/uvx without warming dependency caches",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="warm dependencies for this exact TB2.1 task; repeat as needed",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_ids = set(args.task_id) or None
    if task_ids is not None:
        missing = sorted(task_id for task_id in task_ids if not (TASKS / task_id).is_dir())
        if missing:
            parser.error("unknown Terminal-Bench 2.1 task(s): " + ", ".join(missing))

    prepare_uv_binaries(args.image, args.bin_dir.resolve(), dry_run=args.dry_run)
    if args.binaries_only:
        return 0

    commands = uvx_commands(task_ids)
    pip_installs = pip_commands(task_ids)
    print(
        f"Found {len(commands)} distinct uv verifier dependency sets and "
        f"{len(pip_installs)} pip verifier tasks.",
        flush=True,
    )
    failures: list[str] = []
    for index, (task, uvx) in enumerate(commands, 1):
        docker = ["docker", "run", "--rm"]
        for volume in VOLUMES:
            docker.extend(("--volume", volume))
        docker.extend((args.image, *uvx))
        print(f"[{index}/{len(commands)}] {task}", flush=True)
        if args.dry_run:
            print(shlex.join(docker))
            continue
        result = subprocess.run(docker, check=False)
        if result.returncode:
            failures.append(task)

    for index, (task, image, packages) in enumerate(pip_installs, 1):
        docker = [
            "docker", "run", "--rm", "--volume", PIP_VOLUME,
            "--entrypoint", "pip", image, "install", *packages,
        ]
        print(f"[pip {index}/{len(pip_installs)}] {task}", flush=True)
        if args.dry_run:
            print(shlex.join(docker))
            continue
        result = subprocess.run(docker, check=False)
        if result.returncode:
            failures.append(task)

    if failures:
        print("Failed dependency sets: " + ", ".join(failures))
        return 1
    print("Terminal-Bench 2.1 verifier uv and pip cache prewarm complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
