# PageProbe

MV3 擴充,透過 CDP 操作使用者**既有的分頁**;LLM 經 PageProbeGateway 的 MCP 介面呼叫。

## 操作工具時的原則

- **沒有 navigate**:只能讀 / 操作已開啟的分頁。起手式一律先 `browser_list_tabs` 或
  `browser_get_active_tab` 取得當下 `tabId`(動態配發,不可寫死)。
- **snapshot 優先**:要內容預設用 `browser_snapshot`(CDP accessibility tree,含 `@e1` refs,
  table 以 row/cell 角色呈現)。只在「純文字」用 `browser_get_page_text`、
  「要 href / 完整 HTML」用 `browser_get_page_html`。三者都吃 `selector`。
- **互動靠 ref**:`browser_click` / `fill` / `type` 需要先 snapshot 拿到 ref;ref 綁 `snapshotId`,
  導航後失效,要重新 snapshot。
- **白名單**:origin 不在執行期白名單(`browser_list_allowed_origins`)會 `PERMISSION_DENIED`。

## 真相來源

- 工具參數 / 預設值 / 上限 → `../Protocol/src/commands.ts`(Zod schema)。不要在文件或別處重抄。
- method ↔ MCP tool 對應 → `../PageProbeGateway/src/mcp/tools/register-browser-tools.ts`。

## 詳細文件

架構、資料流、三種取內容方式對照、與 agent-browser 對應、curl 測試範例 →
[`docs/browser-tools.md`](docs/browser-tools.md)。
