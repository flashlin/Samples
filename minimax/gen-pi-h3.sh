#!/usr/bin/env bash
exec bash "$(dirname "${BASH_SOURCE[0]}")/lib/run_python_entry.sh" pi_h3_gen.py "$@"
