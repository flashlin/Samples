# Cursor 設定

Cursor 支援以 `mcp.json` 設定 Streamable HTTP MCP server。

先在啟動 Cursor 的環境設定 token：

```bash
export PAGE_PROBE_TOKEN="<gateway-token>"
```

專案設定放在 `.cursor/mcp.json`，全域設定放在 `~/.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "page-probe": {
      "url": "http://127.0.0.1:17890/mcp",
      "headers": {
        "Authorization": "Bearer ${env:PAGE_PROBE_TOKEN}"
      }
    }
  }
}
```

請勿將實際 token 寫入或提交 `mcp.json`。Cursor 會在 `headers` 中展開 `${env:PAGE_PROBE_TOKEN}`。

重新啟動 Cursor 或重新載入 MCP 設定後，在 MCP 設定或 Agent 的 `Available Tools` 確認 `page-probe` 已連線並公開 browser tools。

官方參考：[Cursor MCP](https://docs.cursor.com/context/model-context-protocol)
