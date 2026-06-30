#!/usr/bin/env bash
set -euo pipefail

# Builds a native lodedb-ffi XCFramework slice for each installed target in
# LODEDB_XCFRAMEWORK_TARGETS. Defaults to the host target for local verification.
# Set LODEDB_XCFRAMEWORK_REQUIRE_ALL=1 to fail when a requested target is missing.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
# The XCFramework is emitted into a stable Artifacts/ directory (not .build/, which
# `swift package clean` wipes) so Package.swift's binaryTarget path is durable.
ARTIFACTS="${ROOT}/Artifacts"
HOST_TARGET="$(rustc -vV | awk '/^host:/ {print $2}')"
TARGETS="${LODEDB_XCFRAMEWORK_TARGETS:-${HOST_TARGET}}"
REQUIRE_ALL="${LODEDB_XCFRAMEWORK_REQUIRE_ALL:-0}"
HEADER_DIR="${REPO_ROOT}/crates/lodedb-ffi/include"
OUTPUT="${ARTIFACTS}/LodeDBCoreFFI.xcframework"

# Pin the static slices' minimum OS to the Package.swift deployment floor so the
# linker does not warn about (and App Store validation does not reject) a mismatch.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-16.0}"

rm -rf "${OUTPUT}"
mkdir -p "${ARTIFACTS}"

INSTALLED_TARGETS="$(rustup target list --installed)"
xcframework_args=()

for target in ${TARGETS}; do
  if ! printf '%s\n' "${INSTALLED_TARGETS}" | grep -qx "${target}"; then
    if [[ "${REQUIRE_ALL}" == "1" ]]; then
      echo "Rust target ${target} is not installed" >&2
      exit 1
    fi
    echo "Skipping Rust target ${target}; install it with: rustup target add ${target}" >&2
    continue
  fi

  cargo build \
    --manifest-path "${REPO_ROOT}/Cargo.toml" \
    -p lodedb-ffi \
    --release \
    --target "${target}"

  library="${REPO_ROOT}/target/${target}/release/liblodedb_ffi.a"
  if [[ ! -f "${library}" ]]; then
    echo "Expected native library was not produced: ${library}" >&2
    exit 1
  fi
  xcframework_args+=(-library "${library}" -headers "${HEADER_DIR}")
done

if [[ "${#xcframework_args[@]}" -eq 0 ]]; then
  echo "No native libraries were built; cannot create XCFramework" >&2
  exit 1
fi

xcodebuild -create-xcframework "${xcframework_args[@]}" -output "${OUTPUT}"
swift build --package-path "${ROOT}" -c release

echo "Created ${OUTPUT}"
