#!/usr/bin/env python3
"""Run, resume, inspect, or regrade an A2E evaluation campaign."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from ageneval.task.orchestrator import CampaignController


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    task_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="campaign YAML to start")
    mode.add_argument("--resume", type=Path, help="existing task/runs campaign directory")
    mode.add_argument("--regrade", type=Path, help="existing campaign directory to regrade")
    parser.add_argument("--grader", action="append", default=[], help="posthoc grader to rerun")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=task_root / "models")
    parser.add_argument("--runs-dir", type=Path, default=task_root / "runs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.grader and args.regrade is None:
        parser.error("--grader requires --regrade")
    if args.rerun_failed and args.resume is None:
        parser.error("--rerun-failed requires --resume")
    if args.dry_run and args.regrade is not None:
        parser.error("--dry-run cannot be combined with --regrade")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)s %(name)s [%(levelname)s] %(message)s",
    )
    task_root = Path(__file__).resolve().parents[1]
    repo_root = task_root.parent
    if args.config is not None:
        controller = CampaignController.from_config(
            args.config,
            runs_directory=args.runs_dir,
            model_directory=args.models_dir,
            repo_root=repo_root,
        )
    else:
        run_directory = args.resume or args.regrade
        controller = CampaignController.from_run_directory(
            run_directory,
            model_directory=args.models_dir,
            repo_root=repo_root,
        )
    controller.prepare()
    log_handler = logging.FileHandler(
        controller.run_directory.root / "campaign.log", encoding="utf-8"
    )
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s %(processName)s %(name)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(log_handler)
    if args.dry_run:
        print(json.dumps(controller.dry_run_summary(), indent=2, ensure_ascii=False))
        return 0
    if args.regrade is not None:
        summary = asyncio.run(controller.regrade(args.grader or None))
    else:
        summary = asyncio.run(controller.run(rerun_failed=args.rerun_failed))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
