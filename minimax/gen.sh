#!/usr/bin/env bash
exec bash "$(dirname "${BASH_SOURCE[0]}")/lib/run_python_entry.sh" sulphur_gen.py "$@"
