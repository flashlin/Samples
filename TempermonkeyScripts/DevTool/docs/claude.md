# Claude Code 設定

Claude Code 可直接加入使用 bearer token 的 Streamable HTTP MCP server。

先設定 token：

```bash
export PAGE_PROBE_TOKEN="<gateway-token>"
```

使用 local scope 加入目前專案：

```bash
claude mcp add \
  --transport http \
  --scope local \
  --header "Authorization: Bearer ${PAGE_PROBE_TOKEN}" \
  page-probe \
  http://127.0.0.1:17890/mcp
```

也可以在專案根目錄使用 `.mcp.json`：

```json
{
  "mcpServers": {
    "page-probe": {
      "type": "http",
      "url": "http://127.0.0.1:17890/mcp",
      "headers": {
        "Authorization": "Bearer ${PAGE_PROBE_TOKEN}"
      }
    }
  }
}
```

請勿將實際 token 寫入或提交 `.mcp.json`。Claude Code 會在啟動時展開 `${PAGE_PROBE_TOKEN}`。

驗證設定：

```bash
claude mcp get page-probe
claude mcp list
```

進入 Claude Code 後執行 `/mcp`，確認 `page-probe` 已連線。專案 scope 的 `.mcp.json` 首次載入時需要核准。

官方參考：[Claude Code MCP](https://code.claude.com/docs/en/mcp)
