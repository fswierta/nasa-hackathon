#!/usr/bin/env bash
# Stop hook: auto-commit + push the agent's work, but NEVER commit anything that looks like an
# API key (the repo is public). If a key is found, nothing is committed or pushed and the
# offending files are named. Keys belong in .env (gitignored), never in .env.example.
#   bash .claude/commit_guard.sh           normal hook run
#   bash .claude/commit_guard.sh --check   only report, change nothing
cd "$(git rev-parse --show-toplevel)" || exit 0
PAT='(AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_.-]{30,}|sk-ant-[0-9A-Za-z_-]{20,}|gsk_[0-9A-Za-z]{20,}|sk-or-[0-9A-Za-z_-]{20,}|_API_KEY[[:space:]]*=[[:space:]]*["'"'"']?[0-9A-Za-z_-]{20,})'

if [ "$1" = "--check" ]; then
  diffcmd() { git diff HEAD -U0 -- "$@"; git ls-files --others --exclude-standard -z | xargs -0 -r cat 2>/dev/null | sed 's/^/+/'; }
  if diffcmd | grep -qE "^\+.*$PAT"; then echo "would BLOCK: key-like text in working tree changes"; exit 1; fi
  echo "ok: no key-like text in changes"; exit 0
fi

git add -A
bad=""
while IFS= read -r -d '' f; do
  if git diff --cached -U0 -- "$f" | grep -qE "^\+.*$PAT"; then bad="$bad $f"; fi
done < <(git diff --cached --name-only -z)

if [ -n "$bad" ]; then
  git reset -q
  echo "BLOCKED auto-commit: possible API key in:$bad — move it to .env (gitignored). Nothing was committed or pushed." >&2
  exit 0
fi

git diff --cached --quiet || git commit -q -m "auto: claude update"
git pull --no-rebase --no-edit --autostash -q origin "$(git branch --show-current)" 2>/dev/null
git push -q -u origin HEAD 2>/dev/null
exit 0
