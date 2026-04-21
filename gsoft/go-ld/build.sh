#!/bin/sh
set -e
cd "$(dirname "$0")/src"
go build -o ../../outputs/ld .
echo "Built: outputs/ld"
