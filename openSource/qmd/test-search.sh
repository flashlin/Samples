#!/bin/bash

# 呼叫 tools/call: search 
echo ">> Calling search tool (Stateless Request)..."
curl -s -X POST http://127.0.0.1:8181/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {
        "query": "如何安裝 Member GRPC SDK? C# 程式碼如何寫?"
      }
    }
  }'
echo
