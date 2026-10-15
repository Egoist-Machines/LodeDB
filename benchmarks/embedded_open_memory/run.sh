#!/usr/bin/env bash
# Builds the memory probe against the working tree (and, with --base, against
# a git revision), then for each store prints the per-stage memory of a
# read-only open, refresh, first and warm search, and checks that both builds
# return identical results (read-only, and after a writable open on a copy).
#
#   benchmarks/embedded_open_memory/run.sh [--base <rev>] [--runs N] <store-dir>...
#
# Stores are never modified; the writable check runs on copies. macOS only.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
base=""
runs=1
stores=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base=$2; shift 2 ;;
    --runs) runs=$2; shift 2 ;;
    *) stores+=("$1"); shift ;;
  esac
done
[ ${#stores[@]} -gt 0 ] || { echo "usage: $0 [--base <rev>] [--runs N] <store-dir>..." >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

build() { # <source-root> <target-dir>
  cargo build --release --quiet --jobs "${JOBS:-4}" \
    --manifest-path "$1/benchmarks/embedded_open_memory/Cargo.toml" --target-dir "$2"
}

builds=(head)
build "$repo" "$repo/target/embedded-open-memory/head"
if [ -n "$base" ]; then
  mkdir -p "$work/base/benchmarks"
  git -C "$repo" archive "$base" Cargo.toml crates third_party | tar -x -C "$work/base"
  cp -R "$here" "$work/base/benchmarks/"
  rm -f "$work/base/benchmarks/embedded_open_memory/Cargo.lock"
  build "$work/base" "$repo/target/embedded-open-memory/base"
  builds=(base head)
fi
bin() { echo "$repo/target/embedded-open-memory/$1/release/$2"; }

for store in "${stores[@]}"; do
  for scenario in "" "--no-refresh" "--mode hybrid --filter"; do
    for build_name in "${builds[@]}"; do
      for _ in $(seq "$runs"); do
        echo "== $(basename "$store") [$build_name] ${scenario:-refresh + vector}"
        # shellcheck disable=SC2086
        "$(bin "$build_name" embedded-open-memory)" "$store" $scenario
      done
    done
  done
  if [ -n "$base" ]; then
    for mode in --refresh --write; do
      for build_name in base head; do
        copy="$work/$build_name-$(basename "$store")"
        rm -rf "$copy"
        cp -R "$store" "$copy"
        chmod -R u+w "$copy"
        "$(bin "$build_name" results_dump)" "$copy" "$mode" > "$work/$build_name.dump"
      done
      if cmp -s "$work/base.dump" "$work/head.dump"; then
        echo "results $(basename "$store") $mode: identical ($(wc -l < "$work/head.dump") lines)"
      else
        echo "results $(basename "$store") $mode: DIFFER"
        diff "$work/base.dump" "$work/head.dump" | head -20
        exit 1
      fi
    done
  fi
done
