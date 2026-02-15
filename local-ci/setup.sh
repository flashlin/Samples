#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "\n${BOLD}=== $1 ===${NC}"; }

print_header "Local CI Setup"

install_gitlab_ci_local() {
    if command -v gitlab-ci-local &>/dev/null; then
        print_status "gitlab-ci-local already installed ($(gitlab-ci-local --version 2>/dev/null || echo 'unknown version'))"
        return 0
    fi

    echo "Installing gitlab-ci-local..."
    if command -v brew &>/dev/null; then
        brew install gitlab-ci-local
    elif command -v npm &>/dev/null; then
        npm install -g gitlab-ci-local
    else
        print_error "Neither brew nor npm found. Please install gitlab-ci-local manually."
        echo "  brew install gitlab-ci-local"
        echo "  OR"
        echo "  npm install -g gitlab-ci-local"
        return 1
    fi
    print_status "gitlab-ci-local installed"
}

check_docker() {
    if command -v docker &>/dev/null && docker info &>/dev/null; then
        print_status "Docker is available"
    else
        print_error "Docker is not available. Please install Docker or OrbStack."
        return 1
    fi
}

check_kubectl() {
    if ! command -v kubectl &>/dev/null; then
        print_error "kubectl not found. Please install kubectl."
        return 1
    fi
    print_status "kubectl is available"

    if kubectl cluster-info &>/dev/null 2>&1; then
        print_status "Kubernetes cluster is reachable"
    else
        print_warning "Kubernetes cluster is not reachable. Deploy jobs will fail."
        print_warning "If using OrbStack, enable Kubernetes in OrbStack settings."
    fi
}

NAMESPACES=(twdc)

ensure_namespaces() {
    for ns in "${NAMESPACES[@]}"; do
        if kubectl get namespace "$ns" &>/dev/null; then
            print_status "Namespace '$ns' already exists"
        else
            kubectl create namespace "$ns"
            print_status "Namespace '$ns' created"
        fi
    done
}

create_global_variables_dir() {
    local global_dir="$HOME/.gitlab-ci-local"
    if [ ! -d "$global_dir" ]; then
        mkdir -p "$global_dir"
        print_status "Created $global_dir"
    else
        print_status "$global_dir already exists"
    fi
}

build_runner_image() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    print_status "Building local CI runner image..."
    docker build -t local-ci-runner:latest -f "$script_dir/Dockerfile.runner" "$script_dir"
    print_status "local-ci-runner:latest image built"
}

install_gitlab_ci_local
check_docker
build_runner_image
check_kubectl
ensure_namespaces
create_global_variables_dir

print_header "Setup Complete"
echo ""
echo "Usage:"
echo "  cd <project-with-gitlab-ci-yml>"
echo "  ../local-ci/local-ci-deploy.sh                        # Interactive mode"
echo "  ../local-ci/local-ci-deploy.sh --list                  # List available jobs"
echo "  ../local-ci/local-ci-deploy.sh --job deploy            # Run specific job"
echo "  ../local-ci/local-ci-deploy.sh -p ../other-project     # Specify project directory"
