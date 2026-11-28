#!/usr/bin/env bash
# Pulls every dataset listed in datasets/manifest.psv into datasets/<tier>/<id>.
# Safe to re-run: datasets that already exist are skipped.
# Git history is dropped to halve disk use; the commit pulled is recorded in datasets/PULLED.psv.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAN="$ROOT/datasets/manifest.psv"
LOG="$ROOT/datasets/PULLED.psv"

while IFS='|' read -r id repo tier license attribution notes; do
  case "$id" in ''|id|\#*) continue ;; esac
  if [ "$repo" = "-" ]; then echo "manual $id (user-supplied, not fetched)"; continue; fi
  dest="$ROOT/datasets/$tier/$id"
  if [ -d "$dest" ]; then echo "skip   $id"; continue; fi
  tmp="$(mktemp -d "$ROOT/datasets/.tmp.XXXXXX")"
  if git clone -q --depth 1 "https://github.com/$repo" "$tmp/r" 2>/dev/null; then
    rev="$(git -C "$tmp/r" rev-parse HEAD)"
    rm -rf "$tmp/r/.git"
    mkdir -p "$ROOT/datasets/$tier"
    mv "$tmp/r" "$dest"
    echo "$id|$repo|$rev" >> "$LOG"
    echo "pulled $id"
  else
    echo "FAILED $id ($repo)"
  fi
  rm -rf "$tmp"
done < "$MAN"
