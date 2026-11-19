#!/usr/bin/env bash
# Starts a roadmap task on a fresh branch from an up-to-date main:
#
#   tools/start_task.sh task-3.1-harness
set --euo pipefail
cd "$(dirname "$0")/.."

name=${1:?usage: tools/start_task.sh BRANCH-NAME}

if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: uncommitted changes; commit or stash them first" >&2
    git status --short >&2
    exit 1
fi
if git show-ref --verify --quiet "refs/heads/$name"; then
    echo "error: a branch named $name already exists; pick another name" >&2
    exit 1
fi

git switch main
git pull --ff-only
git switch -c "$name"
echo "On $name, branched from the latest main. Commit here, then run tools/ship.sh."