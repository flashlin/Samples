# HANDOFF — T1.SqlSharp

> 接手 session 進度交接。本檔**刻意不進 git**。
> 規範:永遠繁中回覆;實作碼禁註解、顯示訊息用英文;**禁用 `git add -A`/`git add .`**(repo root 是上層 `Samples/`,含大量無關 untracked 目錄/二進位),commit 一律明確列路徑;使用者手動 commit,動手前先問。

---

## 當前狀態

### 線一:INSERT/UPDATE ToSql 忠實化 — ✅ 已完成且使用者已 commit

上一輪交接提到「尚未 commit」已過期；使用者明確告知剛剛已 commit。這條線不用再處理。

### 線二:E2E 解析涵蓋率掃描器 — ✅ 方案 A + worker pool 已實作、尚未 commit

使用者選擇方案 A：在 library 新增公開 API，讓逐語句掃描能得到 parser error，而不是沿用會吞錯的 `ExtractStatements()`。

已完成：

- 新增 `SqlParser.ExtractStatementResults()`：逐語句 yield `ParseResult<ISqlExpression>`；成功語句繼續，遇第一個錯誤 yield error 後停止。
- 保留既有 `SqlParser.ExtractStatements()` 行為不變：仍會跳過未知 token 繼續，避免破壞既有 `ExcludeNonSelectStatementTest` 語意。
- 新增 `ExtractStatementResultsTest`，先確認方法不存在紅燈，再實作綠燈。
- 新增 console 專案 `T1.SqlSharpE2eParser`：
  - 預設掃描路徑：`/Users/flash/titan/DbProjects`
  - 可傳參數覆蓋：`dotnet run --project T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj -- <sourcePath> [outputPath]`
  - 遞迴掃描 `*.sql`
  - 長駐 worker pool：固定 N 個 worker process，逐檔送入 worker；若 worker 因 stack overflow 掛掉，主行程記錄該檔失敗並重啟 worker
  - 逐檔 append `report.csv`，中途停止仍保留已處理檔案紀錄
  - 每 100 檔刷新 `summary.json`
  - 完成時輸出完整 `report.json`
  - 預設輸出到 `T1.SqlSharpE2eParser/out/`
  - `out/` 已加入專案根 `.gitignore`
  - parser 失敗是報告資料，程式仍回傳 exit code 0；來源路徑不存在才回傳 1。
- `T1.SqlSharp.sln` 已掛入 `T1.SqlSharpE2eParser`。
- `T1.SqlSharpE2eParser.csproj` 設定 `NuGetAudit=false`，避免離線環境因 vulnerability metadata 查詢產生 `NU1900` warning。
- 真實 corpus 已用 worker pool 完整跑完一次；結果在 `T1.SqlSharpE2eParser/out/`。

### 線三:CREATE PROCEDURE TVP READONLY — ✅ 已完成、尚未 commit

從 partial report 第一個失敗案例定位到：

```sql
CREATE PROCEDURE [dbo].[ArgusJob_InsertM10OverallTurnoverProfile_1.0.0]
    @tvpTable [TvpOverallTurnoverProfile] READONLY
AS
BEGIN
    SET NOCOUNT ON;
END
```

問題：`Parse_ProcedureParameter` 原本不消費 `READONLY`，導致 `CREATE PROCEDURE` 後續讀不到 `AS`，頂層回 `Unknown statement`。

已完成：

- TDD 新增 `ParseCreateProcedureSqlTest.Create_proc_with_table_valued_parameter`
- `SqlProcedureParameter` additive 新增 `IsReadOnly`
- `SqlCreateProcedureStatement.ToSql()` 支援輸出 `READONLY`
- `Parse_ProcedureParameter()` 消費 optional `READONLY`
- 單檔驗證原失敗檔現在 `StatementCount=4`、`SucceededStatements=4`、`FailedStatements=0`

### 線四:report.csv 錯誤分析與 10 個 corpus regression — ✅ 已完成、尚未 commit

已完成：

- `T1.SqlSharpE2eParser` 新增 `--analyze-report [sourcePath] [outputPath]`。
- 已由既有 `out/report.csv` 產出 `T1.SqlSharpE2eParser/out/error.csv` 與 `T1.SqlSharpE2eParser/out/error-summary.csv`。
- `error.csv` 為每個失敗檔一列，包含分類、錯誤訊息、SQL file、offset、line、statement/context preview、suggested test name。
- `error-summary.csv` 為分類彙總，方便挑下一批 TDD 目標。
- 使用者後續要求不要繼續抓 `unknown statement` / 邊界分類，改採隨機抽樣 SQL 檔做 parser regression。
- 新增 `T1.SqlSharpTests/ParseRealCorpusRegressionSqlTest.cs`，10 個測試皆把最小可重現 SQL 內容直接嵌入測試專案，不直接讀 `/Users/flash/titan/DbProjects`。
- 修正 `SqlParser.cs`：
  - procedure parameters 支援 `@param AS DataType`
  - `DECLARE` 支援 `DECLARE @var AS DataType`
  - `WITH EXECUTE AS 'user'` 支援 quoted principal
  - 擴充 `ReservedWords`，避免 `WHILE` / `IF` / `BEGIN` / `END` / `DECLARE` / `RETURN` 等被前一個 `SELECT` 當 alias 吃掉。

10 個 regression 測試來源/語法型態：

- `AccountNotificationAPI_DeleteOldTelegramNotificationLog_21.10.sql`: `SELECT 1` 後接 `WHILE @@ROWCOUNT`
- `Sch_Report_OnlineUserSB.sql`: `DECLARE @x AS INT = 0`
- `m9_dotnet_csmaTotalByMemberx.sql`: procedure parameter `@x AS INT` 與 `AS SELECT`
- `Admin_SB_Pluto_GetRisk1x2TS_6.6.sql`: procedure parameter `@tStamp AS TIMESTAMP`
- `Admin_SB_EventMgmt_AddOddsForNewEvent_1.0.1.sql`: `SELECT @var = @@error` 後接 `IF`
- `Pontus_SB_GetStartedTransaction_4.3.sql`: `IF EXISTS (...) BEGIN SELECT TOP 1 ... ORDER BY`
- `PlutoReplication_RB_TruncateBuffer_RacingBet_14.05.sql`: `WITH EXECUTE AS 'plutoproxy'`
- `AccountAPI_InsertSportsRiskControl_20.05.sql`: TVP `READONLY` + `INSERT ... SELECT ... FROM @tvp`
- `BetGenius_DeleteFixture_1.0.0.sql`: procedure parameter `@fixtureId AS INT` + `DELETE`
- `Aither_LC_Player_VerifyUser_14.02.sql`: `@retcode AS INT OUTPUT` + `RETURN @retcode`

---

## 驗證結果

已通過：

```bash
dotnet test --no-restore
# 622 passed / 0 failed

dotnet build T1.SqlSharp/T1.SqlSharp.csproj --no-restore
# 0 warning / 0 error

dotnet build T1.SqlSharpTests/T1.SqlSharpTests.csproj --no-restore
# 0 warning / 0 error

dotnet build T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj --no-restore
# 0 warning / 0 error
```

這輪 corpus regression 紅燈/綠燈：

```bash
dotnet test T1.SqlSharpTests/T1.SqlSharpTests.csproj --no-restore --filter "FullyQualifiedName~ParseRealCorpusRegressionSqlTest"
# 修改 parser 前: 10 failed / 0 passed
# 修改 parser 後: 10 passed / 0 failed
```

錯誤分析檔產出：

```bash
dotnet run --project T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj --no-build --analyze-report /Users/flash/titan/DbProjects
# Error report: T1.SqlSharpE2eParser/out/error.csv
# Error summary: T1.SqlSharpE2eParser/out/error-summary.csv
# error.csv lines: 23489 (header + 23488 failures)
```

小型臨時 corpus 也已驗證：

```bash
dotnet run --project T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj --no-build -- /private/tmp/t1-sqlsharp-e2e-sample /private/tmp/t1-sqlsharp-e2e-out
# Processed 2/2 | OK 1 | FAIL 1
# JSON report: /private/tmp/t1-sqlsharp-e2e-out/report.json
# CSV report: /private/tmp/t1-sqlsharp-e2e-out/report.csv
```

真實 corpus 完整掃描：

```bash
dotnet run --project T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj --no-build -- /Users/flash/titan/DbProjects
# Processed 46106/46106 | OK 22618 | FAIL 23488
# elapsed: 42.3966562 seconds
# report.csv lines: 46107 (header + 46106 files)
```

`report.json` 摘要：

```text
TotalFiles: 46106
SucceededFiles: 22618
FailedFiles: 23488
TotalStatements: 250043
SucceededStatements: 226555
FailedStatements: 23488
Top ErrorBuckets:
  Unknown statement: 23427
  Result is null: 61
```

TVP 原失敗檔驗證：

```bash
dotnet run --project T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj --no-build --scan-file /Users/flash/titan/DbProjects '/Users/flash/titan/DbProjects/consus.info/Compliance2026/dbo/Stored Procedures/ArgusJob/TurnoverProfile/OverallTurnoverProfile/ArgusJob_InsertM10OverallTurnoverProfile_1.0.0.sql'
# SucceededStatements=4 / FailedStatements=0
```

GitNexus：

```bash
bun /Users/flash/.claude/skills/gitNexus/scripts/detect_changes.ts --repo /Users/flash/vdisk/github/Samples/T1.SqlSharp --scope unstaged --depth 2
# changedSymbolCount=6 / totalUpstream=16
```

已知狀況：

- `dotnet build T1.SqlSharp.sln --no-restore` 會在輸出 `T1.SqlSharp` build 後卡住不退出；已改用逐專案 build 驗證，三個專案皆 0 warning / 0 error。
- 初次 `dotnet restore T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj` 成功，但環境會印 `CSSM_ModuleLoad(): One or more parameters passed to a function were not valid.`；不是 restore 失敗。

---

## 待 commit 路徑

commit 時請明確列路徑，勿用 `git add -A` / `git add .`：

```bash
git add \
  T1.SqlSharp/.gitignore \
  T1.SqlSharp/T1.SqlSharp.sln \
  T1.SqlSharp/T1.SqlSharp/Expressions/SqlCreateProcedureStatement.cs \
  T1.SqlSharp/T1.SqlSharp/Expressions/SqlProcedureParameter.cs \
  T1.SqlSharp/T1.SqlSharp/ParserLit/SqlParser.cs \
  T1.SqlSharp/T1.SqlSharpTests/ParseCreateProcedureSqlTest.cs \
  T1.SqlSharp/T1.SqlSharpTests/ParseRealCorpusRegressionSqlTest.cs \
  T1.SqlSharp/T1.SqlSharpTests/ExtractStatementResultsTest.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj \
  T1.SqlSharp/T1.SqlSharpE2eParser/Program.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/IncrementalScanReportWriter.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/ReportErrorAnalyzer.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/SqlFileWorker.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/SqlCorpusScanner.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/ScanProgress.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/ScanReport.cs \
  T1.SqlSharp/T1.SqlSharpE2eParser/ScanReportWriter.cs
```

建議 commit message：

```text
Add SQL corpus scanner and parser corpus regressions
```

---

## 下一步

1. 使用者確認後 commit 上述路徑。
2. 可直接跑真實 corpus：

```bash
dotnet run --project T1.SqlSharpE2eParser/T1.SqlSharpE2eParser.csproj -- /Users/flash/titan/DbProjects
```

3. 掃描期間可看 `T1.SqlSharpE2eParser/out/report.csv` 與 `summary.json`；完成後看 `report.json` 的 `ErrorBuckets`，再挑 Top failure 分類做下一輪 parser 補洞。

## 參考檔

- 支援現況/待辦:`plans/todo-tsql.md`
- 專案規範與雷點:`CLAUDE.md`
- Parser 主檔:`T1.SqlSharp/ParserLit/SqlParser.cs`
- 測試 helper:`T1.SqlSharpTests/TestHelper.cs`
