#!/usr/bin/env bash
# Ships the current task branch: checks everything locally, pushes, opens the pull request (only
# once), and queues a merge that GitHub performs by itself when every CI check has passed.
#
#   tools/ship.sh
set -euo pipefail
cd "$(dirname "$0")/.."

branch=$(git branch --show-current)
if [[ "$branch" == "main" ]]; then
    echo "error: you are on main, which only accepts pull requests." >&2
    echo "Move your commits to a branch first:" >&2
    echo "  git switch -c <task-branch> && git switch main && git reset --hard origin/main" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: uncommitted changes; commit them first" >&2
    git status --short >&2
    exit 1
fi
git fetch --quiet origin
if [[ -z "$(git log --oneline origin/main..HEAD)" ]]; then
    echo "error: $branch has no commits that main does not already have" >&2
    exit 1
fi

tools/check_local.sh
git push -u origin HEAD
if ! gh pr view --json number > /dev/null 2>&1; then
    gh pr create --base main --fill
fi
gh pr merge --merge --auto

echo
echo "Queued: GitHub merges this pull request as soon as all checks pass."
echo "Follow it with:  gh pr checks --watch"
echo "Afterwards:      git switch main && git pull"