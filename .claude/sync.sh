#!/usr/bin/env bash
# Auto-sync: co INTERVAL sekund pobiera zmiany z GitHuba i scala
# wszystkie zdalne gałęzie (main, pc2, pc3, ...) do bieżącej gałęzi.
# Uruchamia się samo po otwarciu folderu w VS Code (.vscode/tasks.json).
cd "$(dirname "$0")/.." || exit 1
INTERVAL=${1:-10}
echo "Auto-sync włączony (co ${INTERVAL}s). Zamknij ten terminal, żeby wyłączyć."
while true; do
  git fetch -q --prune origin 2>/dev/null
  for b in $(git branch -r --format='%(refname:short)' | grep -vx origin | grep -v HEAD); do
    git merge-base --is-ancestor "$b" HEAD 2>/dev/null && continue
    if git merge -q --no-edit --autostash "$b" >/dev/null 2>&1; then
      echo "$(date +%T)  pobrano zmiany z $b"
    else
      git merge --abort 2>/dev/null
      echo "$(date +%T)  KONFLIKT z $b - pominięto, trzeba scalić ręcznie"
    fi
  done
  sleep "$INTERVAL"
done
