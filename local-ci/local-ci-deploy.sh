#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=""

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[OK]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_header() { echo -e "\n${BOLD}=== $1 ===${NC}\n"; }

JOB_NAME=""
LIST_ONLY=false
DRY_RUN=false
VARIABLES_FILE=""
DEFAULT_IMAGE="local-ci-runner:latest"
USE_SHELL_EXECUTOR=false
EXTRA_ARGS=()

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --job)
                JOB_NAME="$2"
                shift 2
                ;;
            --list)
                LIST_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -p|--project)
                PROJECT_DIR="$(cd "$2" && pwd)"
                shift 2
                ;;
            --variables-file)
                VARIABLES_FILE="$2"
                shift 2
                ;;
            --image)
                DEFAULT_IMAGE="$2"
                shift 2
                ;;
            --shell)
                USE_SHELL_EXECUTOR=true
                shift
                ;;
            -*)
                EXTRA_ARGS+=("$1")
                shift
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

check_gitlab_ci_local() {
    if ! command -v gitlab-ci-local &>/dev/null; then
        print_error "gitlab-ci-local is not installed."
        echo "  Run: ${SCRIPT_DIR}/setup.sh"
        exit 1
    fi
}

check_docker() {
    if ! command -v docker &>/dev/null || ! docker info &>/dev/null 2>&1; then
        print_warning "Docker is not available. Jobs requiring Docker will fail."
    fi
}

check_gitlab_ci_yml() {
    if [ ! -f "${PROJECT_DIR}/.gitlab-ci.yml" ]; then
        print_error ".gitlab-ci.yml not found in ${PROJECT_DIR}"
        echo "  Please run this script from a directory containing .gitlab-ci.yml"
        exit 1
    fi
}

check_kubernetes() {
    if command -v kubectl &>/dev/null && kubectl cluster-info &>/dev/null 2>&1; then
        print_status "Kubernetes cluster is reachable"
    else
        print_warning "Kubernetes cluster is not reachable. Deploy jobs may fail."
    fi
}

check_prerequisites() {
    print_header "Phase 1: Checking prerequisites"
    check_gitlab_ci_local
    print_status "gitlab-ci-local is installed"
    check_docker
    check_gitlab_ci_yml
    print_status ".gitlab-ci.yml found"
    check_kubernetes
}

prompt_clear_cache() {
    local cache_dir="${PROJECT_DIR}/.gitlab-ci-local"
    if [ ! -d "$cache_dir" ]; then
        return
    fi

    read -rp "Clear .gitlab-ci-local cache? (N/y): " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        rm -rf "$cache_dir"
        print_status "Cache directory removed: ${cache_dir}"
    fi
}

resolve_variables_file() {
    if [ -n "$VARIABLES_FILE" ]; then
        if [ ! -f "$VARIABLES_FILE" ]; then
            print_error "Variables file not found: ${VARIABLES_FILE}"
            exit 1
        fi
        VARIABLES_FILE="$(python3 -c "import os; print(os.path.relpath('${VARIABLES_FILE}', '${PROJECT_DIR}'))")"
        return
    fi

    if [ -f "${PROJECT_DIR}/.gitlab-ci-local-variables.yml" ]; then
        print_status "Using variables file: ${PROJECT_DIR}/.gitlab-ci-local-variables.yml"
    fi
}

build_mount_args() {
    local mount_args=()

    if [ -S /var/run/docker.sock ]; then
        mount_args+=("--volume" "/var/run/docker.sock:/var/run/docker.sock")
    fi

    local kubeconfig="${KUBECONFIG:-$HOME/.kube/config}"
    if [ -f "$kubeconfig" ]; then
        mount_args+=("--volume" "${kubeconfig}:/root/.kube/config")
    fi

    echo "${mount_args[@]}"
}

build_ci_variables() {
    local ci_vars=()

    ci_vars+=("--variable" "CI_COMMIT_SHA=$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo 'local')")
    ci_vars+=("--variable" "CI_COMMIT_SHORT_SHA=$(git -C "${PROJECT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'local')")
    ci_vars+=("--variable" "CI_COMMIT_BRANCH=$(git -C "${PROJECT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'local')")
    ci_vars+=("--variable" "CI_PROJECT_NAME=$(basename "${PROJECT_DIR}")")
    ci_vars+=("--variable" "CI_PROJECT_DIR=${PROJECT_DIR}")

    echo "${ci_vars[@]}"
}

setup_environment() {
    print_header "Phase 2: Setting up environment"

    resolve_variables_file
    if [ -n "$VARIABLES_FILE" ]; then
        print_status "Using custom variables file: ${VARIABLES_FILE}"
    elif [ ! -f "${PROJECT_DIR}/.gitlab-ci-local-variables.yml" ]; then
        print_info "No .gitlab-ci-local-variables.yml found (using defaults from .gitlab-ci.yml)"
    fi

    if [ -f "${PROJECT_DIR}/.env" ]; then
        print_status "Found .env file"
    fi
}

list_jobs() {
    print_header "Available Jobs"
    gitlab-ci-local --list --cwd "${PROJECT_DIR_REL}" 2>/dev/null || {
        print_error "Failed to parse .gitlab-ci.yml"
        exit 1
    }
}

select_job_interactive() {
    echo ""
    local jobs_output
    jobs_output=$(gitlab-ci-local --list --cwd "${PROJECT_DIR_REL}" 2>/dev/null) || {
        print_error "Failed to parse .gitlab-ci.yml"
        exit 1
    }

    local job_names=()
    local header_skipped=false
    while IFS= read -r line; do
        if ! $header_skipped; then
            header_skipped=true
            continue
        fi
        local name
        name=$(echo "$line" | awk '{print $1}')
        if [ -n "$name" ]; then
            job_names+=("$name")
        fi
    done <<< "$jobs_output"

    if [ ${#job_names[@]} -eq 0 ]; then
        print_error "No jobs found in .gitlab-ci.yml"
        exit 1
    fi

    if [ ${#job_names[@]} -eq 1 ]; then
        JOB_NAME="${job_names[0]}"
        print_info "Only one job found, auto-selecting: ${JOB_NAME}"
        return
    fi

    echo -e "${BOLD}Select a job to run:${NC}"
    echo ""
    for i in "${!job_names[@]}"; do
        echo "  $((i + 1)). ${job_names[$i]}"
    done
    echo ""

    while true; do
        read -rp "Enter job number (1-${#job_names[@]}): " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#job_names[@]}" ]; then
            JOB_NAME="${job_names[$((choice - 1))]}"
            break
        fi
        echo "Invalid selection. Please enter a number between 1 and ${#job_names[@]}."
    done
}

run_job() {
    local cmd_args=()
    cmd_args+=("--cwd" "${PROJECT_DIR_REL}")

    if [ -n "$VARIABLES_FILE" ]; then
        cmd_args+=("--variables-file" "${VARIABLES_FILE}")
    fi

    local mount_args
    mount_args=$(build_mount_args)
    if [ -n "$mount_args" ]; then
        read -ra mount_array <<< "$mount_args"
        cmd_args+=("${mount_array[@]}")
    fi

    local ci_vars
    ci_vars=$(build_ci_variables)
    read -ra ci_vars_array <<< "$ci_vars"
    cmd_args+=("${ci_vars_array[@]}")

    if $USE_SHELL_EXECUTOR; then
        cmd_args+=("--shell-executor-no-image")
    else
        cmd_args+=("--default-image" "${DEFAULT_IMAGE}")
        cmd_args+=("--no-shell-executor-no-image")
    fi
    cmd_args+=("--network" "host")

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd_args+=("${EXTRA_ARGS[@]}")
    fi

    cmd_args+=("${JOB_NAME}")

    if $DRY_RUN; then
        print_header "Dry Run"
        echo "Would execute:"
        echo "  gitlab-ci-local ${cmd_args[*]}"
        return
    fi

    print_header "Phase 4: Running job '${JOB_NAME}'"
    print_info "Command: gitlab-ci-local ${cmd_args[*]}"
    echo ""

    gitlab-ci-local "${cmd_args[@]}"
}

detect_deploy_job() {
    local job_name="$1"
    local ci_yml="${PROJECT_DIR}/.gitlab-ci.yml"

    local stage
    stage=$(grep -A5 "^${job_name}:" "$ci_yml" 2>/dev/null | grep "stage:" | head -1 | awk '{print $2}' || echo "")

    [ "$stage" = "deploy" ]
}

extract_namespace() {
    local ci_yml="${PROJECT_DIR}/.gitlab-ci.yml"
    local vars_file
    if [ -n "$VARIABLES_FILE" ]; then
        vars_file="${PROJECT_DIR}/${VARIABLES_FILE}"
    else
        vars_file="${PROJECT_DIR}/.gitlab-ci-local-variables.yml"
    fi

    if [ -f "$vars_file" ]; then
        local ns
        ns=$(grep "^NAMESPACE:" "$vars_file" 2>/dev/null | awk '{print $2}' || echo "")
        if [ -n "$ns" ]; then
            echo "$ns"
            return
        fi
    fi

    local ns
    ns=$(grep "NAMESPACE:" "$ci_yml" 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"' || echo "")
    if [ -n "$ns" ]; then
        echo "$ns"
        return
    fi

    local ns_from_script
    ns_from_script=$(grep -o '\-\-namespace [^ ]*' "$ci_yml" 2>/dev/null | head -1 | awk '{print $2}' || echo "")
    if [ -n "$ns_from_script" ]; then
        echo "$ns_from_script"
        return
    fi

    echo "default"
}

verify_deployment() {
    local namespace="$1"

    print_header "Phase 5: Verifying deployment"
    print_info "Namespace: ${namespace}"
    echo ""

    echo -e "${BOLD}Pods:${NC}"
    kubectl get pods -n "$namespace" 2>/dev/null || print_warning "Failed to get pods"
    echo ""

    echo -e "${BOLD}Services:${NC}"
    kubectl get svc -n "$namespace" 2>/dev/null || print_warning "Failed to get services"
    echo ""

    echo -e "${BOLD}PVCs:${NC}"
    kubectl get pvc -n "$namespace" 2>/dev/null || print_warning "Failed to get PVCs"
}

main() {
    parse_arguments "$@"

    if [ -z "$PROJECT_DIR" ]; then
        PROJECT_DIR="$(pwd)"
    fi

    PROJECT_DIR_REL="$(python3 -c "import os; print(os.path.relpath('${PROJECT_DIR}'))")"

    echo -e "${BOLD}Local CI Deploy${NC} - gitlab-ci-local wrapper"
    echo -e "Project: ${CYAN}${PROJECT_DIR}${NC}"

    check_prerequisites
    prompt_clear_cache

    if $LIST_ONLY; then
        list_jobs
        exit 0
    fi

    setup_environment

    if [ -z "$JOB_NAME" ]; then
        print_header "Phase 3: Job selection"
        list_jobs
        select_job_interactive
    fi

    print_info "Selected job: ${JOB_NAME}"

    run_job

    if detect_deploy_job "$JOB_NAME"; then
        local namespace
        namespace=$(extract_namespace)
        verify_deployment "$namespace"
    fi

    echo ""
    print_status "Done!"
}

main "$@"
