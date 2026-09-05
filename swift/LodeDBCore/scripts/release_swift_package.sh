#!/usr/bin/env bash
set -euo pipefail

# Publishes an assembled SwiftPM package to the mirror through a pull request.
#
# Usage:
#   release_swift_package.sh <version> <xcframework_zip_url> <checksum>

version="${1:?usage: release_swift_package.sh <version> <xcframework_zip_url> <checksum>}"
url="${2:?missing xcframework zip url}"
checksum="${3:?missing checksum}"
: "${DEPLOY_TOKEN:?DEPLOY_TOKEN is required}"

if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version must be strict SemVer (X.Y.Z): ${version}" >&2
  exit 2
fi

package_repo="${PACKAGE_REPO:-Egoist-Machines/swift-lodedb}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
branch="release/v${version}"
tag="v${version}"
title="release: LodeDBCore ${version}"
created_workdir=false

if [[ -n "${WORKDIR:-}" ]]; then
  workdir="${WORKDIR}"
  mkdir -p "${workdir}"
else
  workdir="$(mktemp -d)"
  created_workdir=true
fi

clean_url="https://github.com/${package_repo}.git"
clone_url="https://x-access-token:${DEPLOY_TOKEN}@github.com/${package_repo}.git"

cleanup() {
  if [[ -d "${workdir}/.git" ]]; then
    git -C "${workdir}" remote set-url origin "${clean_url}" >/dev/null 2>&1 || true
  fi
  if [[ "${created_workdir}" == true ]]; then
    rm -rf -- "${workdir}"
  fi
}
trap cleanup EXIT

export GH_TOKEN="${DEPLOY_TOKEN}"

git clone "${clone_url}" "${workdir}"

git_user_name="github-actions[bot]"
git_user_email="41898282+github-actions[bot]@users.noreply.github.com"
if identity="$(gh api user --jq '[.login, (.id | tostring)] | @tsv' 2>/dev/null)"; then
  IFS=$'\t' read -r token_login token_user_id <<< "${identity}"
  if [[ -n "${token_login}" && -n "${token_user_id}" ]]; then
    git_user_name="${token_login}"
    git_user_email="${token_user_id}+${token_login}@users.noreply.github.com"
  else
    echo "Warning: token user response was incomplete; using github-actions[bot]." >&2
  fi
else
  echo "Warning: could not resolve the token user; using github-actions[bot]." >&2
fi
git -C "${workdir}" config user.name "${git_user_name}"
git -C "${workdir}" config user.email "${git_user_email}"

remote_main="$(git -C "${workdir}" ls-remote --heads origin refs/heads/main)"
if [[ -z "${remote_main}" ]]; then
  git -C "${workdir}" symbolic-ref HEAD refs/heads/main
  git -C "${workdir}" rm -rfq . 2>/dev/null || true
  "${script_dir}/publish_swift_package.sh" "${workdir}" "${version}" "${url}" "${checksum}"
  git -C "${workdir}" add -A
  assembled_tree="$(git -C "${workdir}" write-tree)"
  git -C "${workdir}" commit -m "${title}"
  release_commit="$(git -C "${workdir}" rev-parse HEAD)"
  git -C "${workdir}" push origin main
  echo "Published initial main at ${release_commit:0:12}"
else
  git -C "${workdir}" checkout -B "${branch}" origin/main
  git -C "${workdir}" rm -rfq . 2>/dev/null || true
  "${script_dir}/publish_swift_package.sh" "${workdir}" "${version}" "${url}" "${checksum}"
  git -C "${workdir}" add -A
  assembled_tree="$(git -C "${workdir}" write-tree)"

  # Check the tag before touching the mirror: a rerun of an already published version
  # must not open a PR that rolls main back, and a changed checksum must fail cleanly.
  git -C "${workdir}" fetch origin --tags
  if git -C "${workdir}" rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    tag_tree="$(git -C "${workdir}" rev-parse "refs/tags/${tag}^{tree}")"
    if [[ "${tag_tree}" != "${assembled_tree}" ]]; then
      echo "Tag ${tag} already exists with a different tree; refusing to move it." >&2
      exit 1
    fi
    echo "Tagged ${tag} (already published)"
    exit 0
  fi

  if git -C "${workdir}" diff --cached --quiet origin/main --; then
    release_commit="$(git -C "${workdir}" rev-parse origin/main)"
    echo "Package content is already on main at ${release_commit:0:12}"
  else
    git -C "${workdir}" commit -m "${title}"
    git -C "${workdir}" push --force origin "${branch}"

    pr_number="$(gh pr list \
      -R "${package_repo}" \
      --base main \
      --head "${branch}" \
      --state open \
      --limit 1 \
      --json number \
      --jq '.[0].number // empty')"
    if [[ -z "${pr_number}" ]]; then
      pr_url="$(gh pr create \
        -R "${package_repo}" \
        --base main \
        --head "${branch}" \
        --title "${title}" \
        --body "Publishes the SwiftPM package for LodeDB ${version}.")"
      pr_number="${pr_url##*/}"
    fi
    if [[ ! "${pr_number}" =~ ^[0-9]+$ ]]; then
      echo "Could not determine the release pull request number." >&2
      exit 1
    fi
    echo "PR #${pr_number}"

    merged=false
    for attempt in {1..6}; do
      if gh pr merge "${pr_number}" \
        -R "${package_repo}" \
        --squash \
        --subject "${title}" \
        --delete-branch; then
        merged=true
        break
      fi
      if (( attempt < 6 )); then
        echo "PR #${pr_number} is not mergeable yet; retrying in 10 seconds." >&2
        sleep 10
      fi
    done
    if [[ "${merged}" != true ]]; then
      echo "Failed to merge PR #${pr_number} after 6 attempts." >&2
      exit 1
    fi

    git -C "${workdir}" fetch origin main
    merged_tree="$(git -C "${workdir}" rev-parse 'origin/main^{tree}')"
    if [[ "${merged_tree}" != "${assembled_tree}" ]]; then
      echo "Merged main tree does not match the assembled package; a concurrent change landed." >&2
      exit 1
    fi
    release_commit="$(git -C "${workdir}" rev-parse origin/main)"
    echo "Merged ${release_commit:0:12}"
  fi
fi

git -C "${workdir}" tag -a "${tag}" -m "LodeDBCore ${version}" "${release_commit}"
git -C "${workdir}" push origin "refs/tags/${tag}"
echo "Tagged ${tag} at ${release_commit:0:12}"
