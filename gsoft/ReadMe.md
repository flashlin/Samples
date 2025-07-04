
Cursor MCP setting
```json
{
  "mcpServers": {
    "browsermcp": {
      "command": "npx",
      "args": ["@browsermcp/mcp@latest"]
    },
    "playwright": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    },
    "interactive-feedback-mcp": {
      "command": "uv",
      "args": [
      "--directory",
      "/Users/flash/vdisk/github/Samples/gsoft/interactive-feedback-mcp",
      "run",
      "server.py"
      ],
      "timeout": 600,
      "autoApprove": ["interactive_feedback"]
    },
    "serena": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/oraios/serena", "serena-mcp-server"]
    },
    "context7-mcp": {                                                                                                                                    
      "command": "docker",
      "args": ["run", "-i", "--rm", "context7-mcp"]
    }
  }
}
```