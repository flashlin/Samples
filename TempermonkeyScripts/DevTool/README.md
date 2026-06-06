# PageProbe

PageProbe 讓 Codex、Claude Code 與 Cursor 透過本機 Streamable HTTP MCP 操作目前已開啟的 Chrome 分頁。

系統由三個 workspace package 組成：

- `Protocol`：Gateway 與 Extension 共用的通訊契約
- `PageProbeGateway`：監聽 `127.0.0.1:17890` 的本機 MCP、HTTP 與 WebSocket Gateway
- `PageProbe`：連接 Gateway 並操作目前 Chrome 分頁的 Manifest V3 Extension

第一版不包含 stdio transport、userscript manager 與 DevTools Panel。

## 環境需求

- Chrome 125 或更新版本
- Bun
- pnpm 10.29.3

所有命令皆從 `DevTool` 目錄執行：

```bash
cd DevTool
pnpm install --frozen-lockfile
pnpm run type-check
pnpm run test
pnpm run test:integration
pnpm run build
```

## 啟動 Gateway

Bearer token 有兩種來源，優先順序如下：

1. `DEVTOOL_TOKEN`
2. `PageProbeGateway/.data/token`

使用明確的環境變數啟動：

```bash
export DEVTOOL_TOKEN="$(openssl rand -hex 32)"
export PAGE_PROBE_TOKEN="$DEVTOOL_TOKEN"
pnpm --filter @page-probe/gateway dev
```

若未設定 `DEVTOOL_TOKEN`，Gateway 會在第一次啟動時產生 token：

```bash
pnpm --filter @page-probe/gateway dev
```

在另一個 terminal 讀取產生的 token：

```bash
cd DevTool
export PAGE_PROBE_TOKEN="$(cat PageProbeGateway/.data/token)"
```

`PAGE_PROBE_TOKEN` 是本文件供 MCP client 與驗證命令使用的環境變數名稱。它的值必須與 Gateway 實際使用的 token 相同。

## 驗證 Gateway

Gateway 啟動後，從另一個 terminal 執行：

```bash
curl --fail-with-body \
  --header "Authorization: Bearer ${PAGE_PROBE_TOKEN}" \
  http://127.0.0.1:17890/health
```

預期結果：

```json
{"status":"ok"}
```

MCP endpoint 為：

```text
http://127.0.0.1:17890/mcp
```

所有 MCP request 都必須包含：

```text
Authorization: Bearer <token>
```

## 載入 Extension

1. 完成 `pnpm run build`。
2. 在 Chrome 開啟 `chrome://extensions`。
3. 啟用 `Developer mode`。
4. 選擇 `Load unpacked`。
5. 載入 `DevTool/PageProbe/.output/chrome-mv3`。
6. 開啟 PageProbe 的 `Extension options`。
7. 將 `Gateway URL` 設為 `ws://127.0.0.1:17890/extension`。
8. 將 `Token` 設為 Gateway 實際使用的 token。
9. 在 `Allowed origins` 每行輸入一個允許控制的完整 origin，例如 `https://example.com`。
10. 只有需要 `browser_evaluate` 時才啟用 `Enable debug evaluation`。
11. 儲存設定並選擇 `Connect`。

## 授權網站

PageProbe 使用 optional host permission，不會在安裝時直接取得所有網站權限。

1. 在 PageProbe Options Page 選擇 `Grant website access`。
2. 在 Chrome 權限提示中允許網站存取。
3. 確認 Options Page 顯示 `Website access granted.`。

Chrome 權限與 PageProbe runtime policy 都必須允許目標網站，PageProbe 才能操作該分頁。

## 設定 MCP Client

設定 client 前，先在啟動 client 的 shell 匯出 token：

```bash
export PAGE_PROBE_TOKEN="<gateway-token>"
```

各 client 的 Streamable HTTP 設定：

- [Codex](docs/codex.md)
- [Claude Code](docs/claude.md)
- [Cursor](docs/cursor.md)

## 啟動順序

1. 啟動 PageProbeGateway。
2. 驗證 `/health`。
3. 載入 PageProbe Extension。
4. 在 Options Page 設定 WebSocket URL 與 token。
5. 授權目標網站。
6. 連接 Extension。
7. 啟動已設定 MCP 的 Codex、Claude Code 或 Cursor。

Gateway、Extension 與 MCP client 必須使用相同 token。
