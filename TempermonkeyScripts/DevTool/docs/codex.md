# Codex 設定

Codex CLI 與 IDE Extension 共用 `config.toml`。可使用全域設定 `~/.codex/config.toml`，或在受信任的專案使用 `.codex/config.toml`。

先在啟動 Codex 的 shell 設定 token：

```bash
export PAGE_PROBE_TOKEN="<gateway-token>"
```

加入以下設定：

```toml
[mcp_servers.page_probe]
url = "http://127.0.0.1:17890/mcp"
bearer_token_env_var = "PAGE_PROBE_TOKEN"
enabled = true
```

`bearer_token_env_var` 會讓 Codex 傳送：

```text
Authorization: Bearer <PAGE_PROBE_TOKEN>
```

重新啟動 Codex 後，在 TUI 執行：

```text
/mcp
```

確認 `page_probe` 已連線並公開 browser tools。

官方參考：[Codex MCP](https://developers.openai.com/codex/mcp)
