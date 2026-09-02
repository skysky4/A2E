#!/usr/bin/env bash
# Push branch yuchenyue to origin via HTTPS + Classic PAT.
# Does NOT save credentials to ~/.git-credentials or git config.
#
# Usage:
#   ./scripts/push_yuchenyue.sh              # prompts for PAT (hidden input)
#   GITHUB_TOKEN=ghp_... ./scripts/push_yuchenyue.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "error: run from inside the AEP git repository" >&2
  exit 1
fi
cd "$ROOT"

BRANCH="yuchenyue"
CURRENT="$(git branch --show-current)"
if [[ "$CURRENT" != "$BRANCH" ]]; then
  echo "error: expected branch '$BRANCH', currently on '$CURRENT'" >&2
  exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  read -rsp "GitHub Classic PAT (ghp_...): " GITHUB_TOKEN
  echo
fi
if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "error: empty token" >&2
  exit 1
fi

ASKPASS="$(mktemp)"
cleanup() {
  rm -f "$ASKPASS"
  unset GITHUB_TOKEN
}
trap cleanup EXIT

cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) echo "YuCYstar" ;;
  *Password*) echo "$GITHUB_TOKEN" ;;
  *) echo "$GITHUB_TOKEN" ;;
esac
EOF
chmod 700 "$ASKPASS"

export GIT_TERMINAL_PROMPT=0
echo "Pushing $BRANCH -> origin/$BRANCH ..."
GIT_ASKPASS="$ASKPASS" git -c credential.helper= push -u origin "$BRANCH"

REMOTE_SHA="$(
  GIT_ASKPASS="$ASKPASS" git -c credential.helper= ls-remote --heads origin "$BRANCH" \
    | awk '{print $1}'
)"
LOCAL_SHA="$(git rev-parse HEAD)"

echo "local:  $LOCAL_SHA"
echo "remote: ${REMOTE_SHA:-<missing>}"
if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  echo "OK: origin/$BRANCH is up to date."
else
  echo "warn: local and remote SHAs differ" >&2
  exit 2
fi
