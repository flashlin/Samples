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