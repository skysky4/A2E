#!/usr/bin/env bash
# Helpers for supervising a run_experiment.py child process.
# This file is sourced by run_tb21_glm53_8_agents.sh.

tb21_database_complete() {
  local python_bin=$1
  local database=$2
  local expected_runs=$3

  [[ -s "$database" ]] || return 1
  "$python_bin" - "$database" "$expected_runs" <<'PY'
import sqlite3
import sys

database, expected_text = sys.argv[1:]
expected = int(expected_text)
try:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise SystemExit(1)
        complete = connection.execute(
            """
            SELECT 1
            FROM experiments AS experiment
            WHERE (
                SELECT COUNT(*)
                FROM experiment_runs AS run
                WHERE run.experiment_id = experiment.id
            ) = ?
            AND (
                SELECT COUNT(*)
                FROM experiment_runs AS run
                JOIN experiment_run_annotations AS annotation
                  ON annotation.experiment_run_id = run.id
                WHERE run.experiment_id = experiment.id
                  AND annotation.name = 'tb_resolved'
            ) = ?
            LIMIT 1
            """,
            (expected, expected),
        ).fetchone()
    finally:
        connection.close()
except sqlite3.Error:
    raise SystemExit(1) from None
raise SystemExit(0 if complete else 1)
PY
}

tb21_terminate_pid() {
  local pid=$1
  local grace_seconds=$2
  local deadline

  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM "$pid" 2>/dev/null || true
  deadline=$((SECONDS + grace_seconds))
  while kill -0 "$pid" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

wait_for_tb21_runner() {
  local pid=$1
  local log_file=$2
  local database=$3
  local expected_runs=$4
  local python_bin=$5
  local exit_grace=$6
  local term_grace=$7
  local marker_seen_at=-1
  local status

  TB21_RUNNER_FORCED_SHUTDOWN=0
  TB21_RUNNER_INVALID_COMPLETION=0

  while kill -0 "$pid" 2>/dev/null; do
    if ((marker_seen_at < 0)) && grep -Fq "experiment finished" "$log_file" 2>/dev/null; then
      marker_seen_at=$SECONDS
    fi
    if ((marker_seen_at >= 0 && SECONDS - marker_seen_at >= exit_grace)); then
      if tb21_database_complete "$python_bin" "$database" "$expected_runs"; then
        TB21_RUNNER_FORCED_SHUTDOWN=1
      else
        TB21_RUNNER_INVALID_COMPLETION=1
      fi
      tb21_terminate_pid "$pid" "$term_grace"
      break
    fi
    sleep 1
  done

  wait "$pid" 2>/dev/null
  status=$?
  if ((TB21_RUNNER_FORCED_SHUTDOWN)); then
    return 0
  fi
  if ((TB21_RUNNER_INVALID_COMPLETION)); then
    return 1
  fi
  return "$status"
}
