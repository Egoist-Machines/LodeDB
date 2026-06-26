#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DERIVED="${ROOT}/.build/xcframework"

rm -rf "${DERIVED}"
mkdir -p "${DERIVED}"

swift build --package-path "${ROOT}" -c release

cat >&2 <<'MSG'
LodeDBCore is currently distributed as a source Swift package.
When the Rust C ABI is packaged for iOS, this script is the hook that will
assemble the Swift wrapper plus native binary into an XCFramework.
MSG
