# PageProbe 瀏覽器工具參考

PageProbe 是 MV3 擴充,透過 Chrome DevTools Protocol (CDP) 操作**使用者既有的分頁**。
LLM 經由 PageProbeGateway 的 MCP 介面呼叫這些工具。

> 重點認知:PageProbe **不自己啟動 / 管理瀏覽器、不開新分頁**,只附著在使用者既有的 Chrome。
> 可用 `browser_navigate` / `browser_reload` 在**既有分頁**內導航,但 `browser_navigate` 的
> 目標 origin 必須在白名單(見「權限模型」)。任何操作的起手式都是先 `browser_list_tabs` 或
> `browser_get_active_tab` 取得當下的 `tabId`(`tabId` 來自 Chrome `chrome.tabs` API,
> 動態配發,關掉重開會變,不能寫死)。

## 架構

monorepo 三套件,以 **Protocol 為單一真相來源**,Gateway 與 Extension 都 import `@page-probe/protocol`:

| 套件 | 角色 | 技術 |
|------|------|------|
| `Protocol` | 三方共用的 Zod 契約(method 名、參數、結果、envelope、錯誤碼) | Zod,輸出型別供雙端共用 |
| `PageProbeGateway` | 對外 MCP server + 對內 WebSocket bridge | Bun (`Bun.serve`) |
| `PageProbe` | MV3 擴充,實際用 CDP 操作分頁 | WXT + chrome.debugger |

請求路徑:

```
LLM / MCP client
  │  HTTP Streamable(Bearer auth + mcp-session-id)
  ▼
PageProbeGateway (Bun)
  ├ http-server ─ bearer-auth ─ request-origin-guard        # 驗證
  ├ mcp-session-store → mcp-server → register-browser-tools  # MCP 工具註冊
  ├ browser-tool-service → command-dispatcher                # 把工具呼叫轉成 protocol 指令
  │                         └ pending-request-store(timeout-policy)
  └ connection-registry ─ extension-connection               # WebSocket + keepAlive 20s ping
  │  WebSocket(protocol envelopes)
  ▼
PageProbe extension(MV3 service worker:entrypoints/background.ts)
  ├ extension-websocket-client      # 自動連線 + 斷線每 10s 重連;狀態寫 chrome.storage.session
  ├ command-processor → permission-service(白名單)→ handler-registry
  ├ services:TabService / PageService / SnapshotService / InteractionService
  │          NetworkCaptureService / DebugEventService / PolicyService
  └ cdp-client ─ tab-session-manager
  │  CDP(chrome.debugger)
  ▼
Chrome 既有分頁
```

關鍵設計點:

- **保活**:MV3 service worker 約 30s 閒置會被驅逐 → Gateway 每 20s 送 `ping` 維持連線;
  Extension 端斷線後每 10s 自動重連。連線狀態(connecting/connected/disconnected)透過
  `chrome.storage.session` 同步給 Options UI 顯示燈號。
- **ref 機制**:`browser_snapshot` 為互動元素配發 `@e1` ref,存進 `RefStore`(綁 `snapshotId` +
  `documentEpoch`);互動工具用 `ref-resolver` 還原成 CDP `backendNodeId`。導航後 ref 失效。
- **權限**:`command-processor` 在執行前過 `permission-service` 白名單檢查(見下)。

## 權限模型

雙層授權,兩層都通過才放行:

1. **執行期白名單**(Extension Options 設定,存於 `chrome.storage`):
   只有清單內的 origin 允許 LLM 讀 / 操作。用 `browser_list_allowed_origins` 查詢。
2. **Chrome host permission**(manifest `<all_urls>`):CDP attach 的前提。

不在白名單的 origin 會回 `PERMISSION_DENIED`。

**導航的雙重檢查**(同一份白名單,不是兩份):`browser_navigate` 除了分頁**當前 origin**
要在白名單(操作既有分頁的前提),**目標 URL 的 origin** 也必須在白名單 —— 否則能把已授權
分頁導去任意網站。同站導航時兩者相同;跨站導航時目標 origin 會被獨立把關。`browser_reload`
不換 URL,只走當前 origin 檢查。

## 取內容的三種方式 —— 怎麼選

| 工具 | 底層機制 | 產出 | 何時用 |
|------|----------|------|--------|
| `browser_snapshot` | CDP `Accessibility.getFullAXTree` | 帶語意角色的無障礙樹 + `@e1` refs | **預設首選**。要「理解結構 + 互動定位」、要 table/list 語意 |
| `browser_get_page_text` | `document.body.innerText`(或 `selector` 的元素) | 渲染後可見純文字 | 只要「人看的文字」、快速擷取內文 |
| `browser_get_page_html` | `outerHTML` / `selector` 的 `innerHTML` | HTML 原始結構(預設清掉 script/style/on* 屬性) | 要保留**連結 href**、複雜 table 結構、自行解析 DOM |

### 「snapshot 優先」原則

對 AI 而言,**`browser_snapshot` 是預設首選**,因為它給的不是一坨文字,而是帶
`role` / `name` / `ref` 的結構化樹:

- table 會以 `table / row / cell / columnheader / rowheader` 角色呈現(有語意,不靠排版猜)。
- 互動元素(button/link/textbox…)會配 `@e1` 之類的 ref,可直接餵給
  `browser_click` / `browser_fill` / `browser_type` 做**確定性定位**。
- 只有在「純粹要內文文字」或「要完整 HTML / href」時,才退回 `get_page_text` / `get_page_html`。

## 同一張 table 的三種輸出對照

來源:`https://www.tangpc.com.tw/blog/detail/1341` 的記憶體價格表。

### `browser_snapshot`(accessibility tree)

```
- row
  - cell "UMAX 16GB DDR5-4800"
  - cell "$1,039"
  - cell "$4,750"
  - cell "暴漲約 4.5 倍"
  ...
```

角色齊全:`table` / `row` / `cell` / `columnheader` / `rowheader` / `rowgroup`。
適合 LLM 理解欄列關係。

### `browser_get_page_text`(innerText)

同列儲存格用 `\t` 分隔、列與列用 `\n` 分隔:

```
品牌規格\t114年07月價格 (約)\t115年02月價格 (約)\t漲幅趨勢
UMAX 16GB DDR5-4800\t$1,039\t$4,750\t暴漲約 4.5 倍
```

可 `split("\t")` 還原欄位,但**會遺失**:合併儲存格(colspan/rowspan)、thead/tbody
角色、儲存格內 href / 圖片、CSS 隱藏欄。

### `browser_get_page_html`(outerHTML)

保留 `<table><tr><td>` 完整結構與屬性(含 `colspan`、連結 href),需要精準還原時用。

## 工具清單(19 個)

> **精確參數 / 預設值 / 上限以 `Protocol/src/commands.ts` 的 Zod schema 為單一真相來源**,
> 此處不重列以免過時。透過 MCP 呼叫時,`tools/list` 回應本來就帶每個工具的 `inputSchema`。
> 共通慣例:`tabId` 一律必填且動態取得;有 `maxChars` / `maxBytes` 的工具超量時回 `truncated: true`。

**分頁**
- `browser_list_tabs` — 列出所有分頁(`id` / `title` / `url` / `active` / `status`)
- `browser_get_active_tab` — 取目前作用中分頁

**頁面內容**
- `browser_get_page_metadata` — `title` / `url` / `origin`
- `browser_snapshot` — 無障礙樹 + refs;`selector` 走 `getPartialAXTree` 限縮範圍(**首選**)
- `browser_get_page_text` — `innerText`,可帶 `selector`
- `browser_get_page_html` — `outerHTML` / `innerHTML`,可帶 `selector`,預設 sanitize

**導航**(在既有分頁內,不開新分頁)
- `browser_navigate` — 將既有分頁導向新 URL(CDP `Page.navigate`);**目標 origin 也須在白名單**
- `browser_reload` — 重新整理分頁(CDP `Page.reload`,可帶 `ignoreCache`)

**互動**(需先 snapshot 取得 ref;ref 綁 `snapshotId`,導航後失效)
- `browser_click` / `browser_fill` / `browser_type` / `browser_get_element_text`

**網路**
- `browser_start_network_capture` / `browser_list_network_requests` / `browser_get_response_body`

**偵錯**
- `browser_get_console` / `browser_get_errors`
- `browser_evaluate` — 在分頁執行 JS,受 `debugEvaluateEnabled` 設定管控

**政策**
- `browser_list_allowed_origins` — 回傳執行期白名單 `{ origins: string[] }`

## 與 agent-browser 的對應

兩者底層機制其實一致(都走 CDP),只是 CLI vs MCP 介面差異:

| agent-browser | PageProbe | 共同底層 |
|---------------|-----------|----------|
| `snapshot`(`-i`/`-c`/`-d`/`-s`/`--urls`) | `browser_snapshot`(`interactiveOnly`/`compact`/`maxDepth`/`selector`/`includeUrls`) | `Accessibility.getFullAXTree` / `getPartialAXTree` |
| `get text <sel>` | `browser_get_page_text`(`selector`) | `innerText`(agent-browser fallback `textContent`) |
| `get html <sel>` | `browser_get_page_html`(`selector`) | `innerHTML` / `outerHTML` |
| `click @ref` | `browser_click` | CDP backendNodeId |
| `open` / `goto` / `navigate <url>` | `browser_navigate` | CDP `Page.navigate` |
| `reload` | `browser_reload` | CDP `Page.reload` |
| `back` / `forward` | (未提供) | agent-browser 用 `history.back/forward` |

agent-browser 原始碼佐證:`cli/src/native/snapshot.rs` 用 `Accessibility.getFullAXTree`;
`cli/src/native/element.rs` 的 `get text` 是 `this.innerText || this.textContent`。

## 已知限制

- **不開新分頁、無前進 / 後退**:可在既有分頁 `browser_navigate` / `browser_reload`,
  但不會自己開新分頁,也未提供 back / forward。
- **白名單**:origin 不在執行期白名單會 `PERMISSION_DENIED`;`browser_navigate` 連目標 origin 也要過白名單。
- `innerText` **不含**:連結 href、圖片 alt、`display:none` 隱藏元素、跨 iframe 內容、尚未 lazy-load 的區塊。
- ref 綁 `snapshotId`,導航後失效。
- `browser_snapshot` 的 `includeIframes` / `includeCursorInteractive` 已在 schema 保留,但 service 端尚未實作。

## 直接用 curl 測 gateway

```bash
TOKEN=$(cat PageProbeGateway/.data/token); URL=http://127.0.0.1:17890/mcp
ACC="application/json, text/event-stream"

# 1. initialize → 取 mcp-session-id (response header)
curl -s -D - -o /dev/null "$URL" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -H "Accept: $ACC" -H "mcp-protocol-version: 2025-03-26" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1.0.0"}}}'

# 2. notifications/initialized (帶 mcp-session-id)
# 3. tools/call,例如:
#    {"jsonrpc":"2.0","id":2,"method":"tools/call",
#     "params":{"name":"browser_snapshot","arguments":{"tabId":123,"maxChars":1000000}}}
```

回應是 SSE,內容在 `data:` 行;工具結果在 `result.content[0].text`(是一段 JSON 字串,需再解析)。
