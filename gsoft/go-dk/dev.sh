#!/bin/bash
export PATH="/usr/local/go/bin:$PATH"
go build -o dk . && cp dk ../ && echo "Build OK"
