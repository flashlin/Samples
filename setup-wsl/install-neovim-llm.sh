#!/bin/bash
set -e

# ref: https://ollama.ai/library/codellama
curl https://ollama.ai/install.sh | sh
ollama run mistral:instruct "Write C# code that output 'Hello World' string"
# ollama run codellama:7b "Write me a function that outputs the fibonacci sequence"