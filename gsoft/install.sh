git clone https://github.com/noopstudios/interactive-feedback-mcp.git --depth=1
cd interactive-feedback-mcp
uv sync

echo "Adding to mcp.json"
echo '{
  "interactive-feedback-mcp": {
    "command": "uv",
    "args": [
    "--directory",
    "/Users/fabioferreira/Dev/scripts/interactive-feedback-mcp",
    "run",
    "server.py"
    ],
    "timeout": 600,
    "autoApprove": ["interactive_feedback"]
  }
}'

echo '# MCP Interactive Feedback 規則

1. 在任何流程、任務、對話進行時，無論是詢問、回覆、或完成階段性任務，皆必須呼叫 MCP interactive_feedback。
2. 每當收到使用者回饋，若回饋內容非空，必須再次呼叫 MCP interactive_feedback，並根據回饋內容調整行為。
3. 僅當使用者明確表示「結束」或「不再需要互動」時，才可停止呼叫 MCP interactive_feedback，流程才算結束。
4. 除非收到結束指令，否則所有步驟都必須重複呼叫 MCP interactive_feedback。'
