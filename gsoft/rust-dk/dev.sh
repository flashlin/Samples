#!/bin/bash

# dk Docker Management Tool - Development Script
# Quick compile and run script for development

set -e

echo "🔧 dk Development Script"
echo "======================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    print_error "Rust/Cargo not found. Please install Rust first."
    echo "Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Source cargo environment if needed
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

print_status "Checking Rust version..."
cargo --version

# Check if Docker is available (optional warning)
if ! command -v docker &> /dev/null; then
    print_warning "Docker not found. dk will work but won't be able to manage containers."
    print_warning "Install Docker from: https://docs.docker.com/get-docker/"
else
    print_success "Docker found: $(docker --version)"
fi

# Parse command line arguments
BUILD_TYPE="debug"
RUN_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --release|-r)
            BUILD_TYPE="release"
            shift
            ;;
        --clean|-c)
            print_status "Cleaning previous builds..."
            cargo clean
            shift
            ;;
        --check)
            print_status "Running cargo check..."
            cargo check
            exit 0
            ;;
        --test|-t)
            print_status "Running tests..."
            cargo test
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [-- DK_ARGS]"
            echo ""
            echo "Options:"
            echo "  --release, -r     Build in release mode"
            echo "  --clean, -c       Clean before building"
            echo "  --check          Run cargo check only"
            echo "  --test, -t       Run tests only"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                # Build debug and run dk"
            echo "  $0 --release      # Build release and run dk"
            echo "  $0 -- logs        # Build and run 'dk logs'"
            echo "  $0 -- --help      # Build and run 'dk --help'"
            exit 0
            ;;
        --)
            shift
            RUN_ARGS="$@"
            break
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the project
if [ "$BUILD_TYPE" = "release" ]; then
    print_status "Building dk in release mode..."
    cargo build --release
    BINARY_PATH="./target/release/dk"
else
    print_status "Building dk in debug mode..."
    cargo build
    BINARY_PATH="./target/debug/dk"
fi

# Check if build was successful
if [ $? -eq 0 ]; then
    print_success "Build completed successfully!"
else
    print_error "Build failed!"
    exit 1
fi

# Check if binary exists and is executable
if [ ! -f "$BINARY_PATH" ]; then
    print_error "Binary not found at $BINARY_PATH"
    exit 1
fi

# Make sure binary is executable
chmod +x "$BINARY_PATH"

print_status "Binary location: $BINARY_PATH"
print_status "Binary size: $(du -h "$BINARY_PATH" | cut -f1)"

# Run the binary
echo ""
print_status "Running dk $RUN_ARGS"
echo "========================"

# Execute dk with any provided arguments
if [ -n "$RUN_ARGS" ]; then
    exec "$BINARY_PATH" $RUN_ARGS
else
    exec "$BINARY_PATH"
fi