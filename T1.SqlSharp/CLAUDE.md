# T1.SqlSharp — 專案指引

手寫 recursive-descent T-SQL parser。入口 `SqlParser.Parse()` dispatch 頂層語句
（WITH CTE / CREATE TABLE / SELECT / EXEC sp_addextendedproperty / SET）。
支援現況清單見 `plans/todo-tsql.md`；接手 session 進度見 `HANDOFF.md`（若存在）。

## 檔案地圖

- Parser：`T1.SqlSharp/ParserLit/SqlParser.cs`（主，約 4000 行）、`LinqParser.cs`（LINQ → SQL）
- AST：`T1.SqlSharp/Expressions/*.cs`
  - `SqlType.cs`：每個 AST 類別對應一個 enum 成員
  - `SqlVisitor.cs`：走訪邏輯，每類別一個 `Visit_Xxx`
- 測試：`T1.SqlSharpTests/*.cs`，helper 在 `TestHelper.cs`

## 新增一個 T-SQL 語法的 TDD recipe（必照）

1. **型別表面先行**（讓測試能編譯，behavior 仍紅）：
   - 新 AST 類別 → 同步三處：`SqlType` enum 加成員、`SqlVisitor` 加 `Visit_Xxx`、類別 `Accept` 呼叫它
   - 必填參考型別用 `required`；字串 `= string.Empty`、集合 `= []`、`TextSpan = new()`
   - PATH/AUTO 之類「只差一個 mode」的變體用單一類別 + enum，**不要**複製成多個類別（避免 Duplicated Code）
2. 寫測試 → 跑 → **確認紅燈**（且紅的原因要對：parser 尚未解析該語法）
3. 實作 parser → 跑 → **綠燈**
4. 跑完整套件確認零回歸 + `dotnet build` 確認 0 警告
5. 打勾 `plans/todo-tsql.md`（`[x]`/`[~]`）並更新「最後驗證」日期

## 測試慣例（NUnit + FluentAssertions 6.12.1）

- `sql.ParseSql()` 取 `ParseResult<ISqlExpression>`，再 `rc.ShouldBe(new SelectStatement { ... })`
- `ShouldBe`/`ShouldBeList`（`TestHelper.cs`）自動排除 `Span`，用 RespectingRuntimeTypes + ExcludingMissingMembers
- 平面值 → `SqlFieldExpr` / `SqlValue{ SqlType=IntValue }`；整數走 `ParseArithmeticExpr`
- 引號字串值含引號本身（如 `RootName = new SqlValue { Value = "'customer'" }`）

## 雷點（踩過，別重蹈）

1. **`ReservedWords`（`SqlParser.cs`）漏列 → 關鍵字被當別名吃掉**：
   新子句關鍵字（ORDER/HAVING/INTERSECT…）未列入，`FROM customer <KW>` 會把 `<KW>` 當 table 別名。
   欄位別名排除在 `Parse_SelectColumns`（`!IsPeekKeywords("FROM") && !IsPeekKeywords("INTO")`）。
   加新子句關鍵字時記得評估是否要加進這兩處。
2. **`dotnet test`/`restore` 會打私有 feed `nugetv3.coreop.net`（離線連不到）**：
   已由 repo 根 `nuget.config`（`<clear/>` + 只留 nuget.org）解掉，套件靠本機全域快取。
   換機器且私有套件 `T1.Standard` 不在快取時 restore 會失敗——已知風險。
3. **SqlVisitor 覆蓋缺口（限制非 bug）**：`SqlInnerTableSource`（衍生表 `FROM (SELECT…) t`）
   未覆寫 `Accept`，不會走訪內部子查詢；想測括號運算式走「欄位位置的 `(CASE…)`」這種會被走訪的路徑。
4. **`OnCondition` 為 nullable**（CROSS JOIN / APPLY 無 ON）：走訪/輸出 join 條件處要 null-safe。

## 常用指令

```bash
dotnet test                                              # 全套件（自動 build）
dotnet test --filter "FullyQualifiedName~ForJson"        # 只跑某組
dotnet build T1.SqlSharp/T1.SqlSharp.csproj --no-incremental 2>&1 | grep -i warning  # 確認 0 警告
```
