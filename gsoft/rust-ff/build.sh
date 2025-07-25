#!/bin/bash

# Build script for Rust project
echo "Building Rust project..."

# Run cargo build in release mode
cargo build --release

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
else
    echo "Build failed!"
    exit 1
fi

cargo run --release
