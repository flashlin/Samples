#!/usr/bin/env bash
set -euo pipefail

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

SYSTEM_NAMESPACES=("kube-system" "kube-public" "kube-node-lease" "default")
FORCE=false
INCLUDE_SYSTEM=false
DELETE_MODE=false
TARGET_NAMESPACE=""
TOTAL_SCALED=0
TOTAL_DELETED=0

show_help() {
    echo -e "${BOLD}shutdown.sh${NC} - Stop all resources in local K8s cluster"
    echo ""
    echo "Usage: shutdown.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --force            Skip confirmation prompt"
    echo "  --all              Include system namespaces"
    echo "  --delete           Delete resources instead of scaling to 0"
    echo "  --namespace NAME   Only target the specified namespace"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  shutdown.sh                          Interactive with system ns excluded"
    echo "  shutdown.sh --force                  Skip confirmation"
    echo "  shutdown.sh --namespace twdc         Only target 'twdc' namespace"
    echo "  shutdown.sh --force --delete         Force delete all resources"
    echo "  shutdown.sh --all                    Include system namespaces"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE=true
                shift
                ;;
            --all)
                INCLUDE_SYSTEM=true
                shift
                ;;
            --delete)
                DELETE_MODE=true
                shift
                ;;
            --namespace)
                TARGET_NAMESPACE="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

check_kubectl() {
    print_header "Checking prerequisites"

    if ! command -v kubectl &>/dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    print_status "kubectl is installed"

    if ! kubectl cluster-info &>/dev/null 2>&1; then
        print_error "Kubernetes cluster is not reachable"
        exit 1
    fi
    print_status "Kubernetes cluster is reachable"
}

is_system_namespace() {
    local ns="$1"
    for sys_ns in "${SYSTEM_NAMESPACES[@]}"; do
        if [[ "$ns" == "$sys_ns" ]]; then
            return 0
        fi
    done
    return 1
}

collect_target_namespaces() {
    local namespaces=()

    if [[ -n "$TARGET_NAMESPACE" ]]; then
        if ! kubectl get namespace "$TARGET_NAMESPACE" &>/dev/null 2>&1; then
            print_error "Namespace '${TARGET_NAMESPACE}' not found"
            exit 1
        fi
        namespaces+=("$TARGET_NAMESPACE")
        echo "${namespaces[@]}"
        return
    fi

    local all_ns
    all_ns=$(kubectl get namespaces -o jsonpath='{.items[*].metadata.name}')

    for ns in $all_ns; do
        if ! $INCLUDE_SYSTEM && is_system_namespace "$ns"; then
            continue
        fi
        namespaces+=("$ns")
    done

    if [[ ${#namespaces[@]} -eq 0 ]]; then
        print_warning "No target namespaces found"
        exit 0
    fi

    echo "${namespaces[@]}"
}

count_resources() {
    local ns="$1"
    local resource="$2"
    kubectl get "$resource" -n "$ns" --no-headers 2>/dev/null | wc -l | tr -d ' '
}

display_shutdown_plan() {
    local namespaces=("$@")

    print_header "Shutdown plan"

    if $DELETE_MODE; then
        print_warning "Mode: DELETE (resources will be permanently removed)"
    else
        print_info "Mode: SCALE TO ZERO (reversible)"
    fi
    echo ""

    for ns in "${namespaces[@]}"; do
        local deploy_count statefulset_count rs_count pod_count
        deploy_count=$(count_resources "$ns" "deployments")
        statefulset_count=$(count_resources "$ns" "statefulsets")
        rs_count=$(count_resources "$ns" "replicasets")
        pod_count=$(count_resources "$ns" "pods")

        echo -e "  ${BOLD}${ns}${NC}: ${deploy_count} deployments, ${statefulset_count} statefulsets, ${rs_count} replicasets, ${pod_count} pods"
    done
    echo ""
}

confirm_shutdown() {
    if $FORCE; then
        return
    fi

    read -rp "Proceed with shutdown? (y/N): " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        print_info "Shutdown cancelled"
        exit 0
    fi
}

scale_deployments() {
    local ns="$1"
    local count
    count=$(count_resources "$ns" "deployments")

    if [[ "$count" -eq 0 ]]; then
        return
    fi

    if $DELETE_MODE; then
        kubectl delete deployments --all -n "$ns" --grace-period=30 2>/dev/null
        print_status "Deleted ${count} deployments in ${ns}"
    else
        kubectl scale deployments --all --replicas=0 -n "$ns" 2>/dev/null
        print_status "Scaled ${count} deployments to 0 in ${ns}"
    fi
    TOTAL_SCALED=$((TOTAL_SCALED + count))
}

scale_statefulsets() {
    local ns="$1"
    local count
    count=$(count_resources "$ns" "statefulsets")

    if [[ "$count" -eq 0 ]]; then
        return
    fi

    if $DELETE_MODE; then
        kubectl delete statefulsets --all -n "$ns" --grace-period=30 2>/dev/null
        print_status "Deleted ${count} statefulsets in ${ns}"
    else
        kubectl scale statefulsets --all --replicas=0 -n "$ns" 2>/dev/null
        print_status "Scaled ${count} statefulsets to 0 in ${ns}"
    fi
    TOTAL_SCALED=$((TOTAL_SCALED + count))
}

scale_replicasets() {
    local ns="$1"
    local count
    count=$(count_resources "$ns" "replicasets")

    if [[ "$count" -eq 0 ]]; then
        return
    fi

    if $DELETE_MODE; then
        kubectl delete replicasets --all -n "$ns" --grace-period=30 2>/dev/null
        print_status "Deleted ${count} replicasets in ${ns}"
    else
        kubectl scale replicasets --all --replicas=0 -n "$ns" 2>/dev/null
        print_status "Scaled ${count} replicasets to 0 in ${ns}"
    fi
    TOTAL_SCALED=$((TOTAL_SCALED + count))
}

delete_pods() {
    local ns="$1"
    local count
    count=$(count_resources "$ns" "pods")

    if [[ "$count" -eq 0 ]]; then
        return
    fi

    kubectl delete pods --all -n "$ns" --grace-period=30 2>/dev/null
    print_status "Deleted ${count} pods in ${ns}"
    TOTAL_DELETED=$((TOTAL_DELETED + count))
}

shutdown_namespace() {
    local ns="$1"
    print_header "Shutting down namespace: ${ns}"

    scale_deployments "$ns"
    scale_statefulsets "$ns"
    scale_replicasets "$ns"
    delete_pods "$ns"
}

shutdown_all_namespaces() {
    local namespaces=("$@")

    for ns in "${namespaces[@]}"; do
        shutdown_namespace "$ns"
    done
}

print_summary() {
    print_header "Summary"

    if $DELETE_MODE; then
        print_status "Resources deleted: ${TOTAL_SCALED}"
    else
        print_status "Resources scaled to 0: ${TOTAL_SCALED}"
    fi
    print_status "Pods deleted: ${TOTAL_DELETED}"
    echo ""
    print_status "Shutdown complete!"
}

main() {
    parse_arguments "$@"

    echo -e "${BOLD}Local CI Shutdown${NC} - Stop K8s cluster resources"

    check_kubectl

    local ns_list
    ns_list=$(collect_target_namespaces)

    local namespaces
    read -ra namespaces <<< "$ns_list"

    display_shutdown_plan "${namespaces[@]}"
    confirm_shutdown
    shutdown_all_namespaces "${namespaces[@]}"
    print_summary
}

main "$@"
