#!/usr/bin/env bash
set -euo pipefail

readonly RETENTION_DAYS=30
readonly SCAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: ${0##*/} [--force]"
  echo ""
  echo "Removes build output directories (node_modules, bin, obj) whose project"
  echo "source has not changed in the last ${RETENTION_DAYS} days."
  echo ""
  echo "Manifests and lock files are ignored when measuring activity, because"
  echo "batch dependency updates touch them without real development work."
  echo "A directory is only considered a build output when its parent looks like"
  echo "a matching project, so directories such as .venv/bin are left alone."
  echo ""
  echo "Dry-run unless --force is given."
}

find_candidate_dirs() {
  find "$SCAN_ROOT" -type d \
    \( -name node_modules -o -name bin -o -name obj \) -prune -print 2>/dev/null | sort
}

has_file_matching() {
  local dir="$1"
  shift
  local pattern
  for pattern in "$@"; do
    if [[ -n "$(find "$dir" -maxdepth 1 -type f -name "$pattern" -print -quit 2>/dev/null)" ]]; then
      return 0
    fi
  done
  return 1
}

is_node_modules_output() {
  has_file_matching "$(dirname "$1")" 'package.json'
}

is_dotnet_output() {
  local artifact_dir="$1"
  if has_file_matching "$(dirname "$artifact_dir")" \
      '*.csproj' '*.vbproj' '*.fsproj' '*.vcxproj'; then
    return 0
  fi
  [[ "$(basename "$artifact_dir")" == "obj" ]] &&
    has_file_matching "$artifact_dir" 'project.assets.json' '*.nuget.g.props'
}

is_build_output() {
  local artifact_dir="$1"
  case "$(basename "$artifact_dir")" in
    node_modules) is_node_modules_output "$artifact_dir" ;;
    bin | obj) is_dotnet_output "$artifact_dir" ;;
    *) return 1 ;;
  esac
}

latest_source_mtime() {
  local project_dir="$1"
  local mtime
  mtime=$(find "$project_dir" \
    -type d \( -name node_modules -o -name .git -o -name bin -o -name obj \
               -o -name dist -o -name build -o -name .next -o -name .nuxt \
               -o -name out -o -name coverage -o -name .venv \) -prune -o \
    -type f \
      ! -name 'package.json' \
      ! -name 'package-lock.json' \
      ! -name 'pnpm-lock.yaml' \
      ! -name 'yarn.lock' \
      ! -name '.DS_Store' \
      -exec stat -f '%m' {} + 2>/dev/null | sort -rn | head -1)
  echo "${mtime:-0}"
}

directory_size() {
  du -sh "$1" 2>/dev/null | cut -f1
}

format_date() {
  if (( $1 == 0 )); then
    echo "unknown   "
  else
    date -r "$1" '+%Y-%m-%d'
  fi
}

classify_candidates() {
  local cutoff="$1"
  local artifact_dir project_dir mtime status

  while IFS= read -r artifact_dir; do
    [[ -n "$artifact_dir" ]] || continue
    project_dir="$(dirname "$artifact_dir")"

    if ! is_build_output "$artifact_dir"; then
      printf 'SKIP\t%-6s %-34s %s\n' "SKIP" "not a build output" \
        "${artifact_dir#"$SCAN_ROOT"/}"
      continue
    fi

    mtime="$(latest_source_mtime "$project_dir")"
    if (( mtime > cutoff )); then
      status="KEEP"
    else
      status="REMOVE"
    fi

    printf '%s\t%-6s last-change=%s  size=%-6s %s\t%s\n' \
      "$status" "$status" "$(format_date "$mtime")" \
      "$(directory_size "$artifact_dir")" \
      "${artifact_dir#"$SCAN_ROOT"/}" "$artifact_dir"
  done < <(find_candidate_dirs)
}

remove_directories() {
  local dir
  for dir in "$@"; do
    echo "Removing ${dir#"$SCAN_ROOT"/}"
    rm -rf "$dir"
  done
  echo "Removed $# directories."
}

main() {
  local force=false

  case "${1:-}" in
    --force) force=true ;;
    "") ;;
    *) usage; exit 1 ;;
  esac

  local cutoff
  cutoff=$(date -v-"${RETENTION_DAYS}"d +%s)

  echo "Scanning ${SCAN_ROOT}"
  echo "Retention: ${RETENTION_DAYS} days (cutoff $(format_date "$cutoff"))"
  echo ""

  local stale_dirs=()
  local status display path
  while IFS=$'\t' read -r status display path; do
    echo "$display"
    if [[ "$status" == "REMOVE" ]]; then
      stale_dirs+=("$path")
    fi
  done < <(classify_candidates "$cutoff")

  echo ""
  if (( ${#stale_dirs[@]} == 0 )); then
    echo "Nothing to remove."
    exit 0
  fi

  if [[ "$force" == true ]]; then
    remove_directories "${stale_dirs[@]}"
  else
    echo "Dry run: ${#stale_dirs[@]} directories would be removed."
    echo "Re-run with --force to delete them."
  fi
}

main "$@"
