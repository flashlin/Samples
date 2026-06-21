# HANDOFF — T1.SqlSharp T-SQL Parser 擴充

> 最後更新：2026-06-21
> 接手方式：在專案根目錄 `/Users/flash/vdisk/github/Samples/T1.SqlSharp` 開新對話，附上本檔路徑即可。

---

## Goal（目標）

逐步為 `T1.SqlSharp` 的 T-SQL parser 補齊缺少的語法，**一律用 TDD**（先紅燈、後綠燈、零回歸），
並以 `plans/todo-tsql.md` 作為待辦清單：每完成一項就把對應 `[ ]` 改成 `[x]`（部分支援 `[~]`、不適用 T-SQL `[N/A]`），更新「最後驗證」日期。

> **長存的開發規則 / 雷點 / 檔案地圖已抽到專案根 `CLAUDE.md`**（每 session 自動載入）。本檔只記「當前進度 + 下一步」這類會過期的 session 狀態，不重複 CLAUDE.md 的內容。

專案本質：手寫 recursive-descent T-SQL parser。入口 `SqlParser.Parse()`。AST 在 `T1.SqlSharp/Expressions/`，
parser 在 `T1.SqlSharp/ParserLit/SqlParser.cs` + `LinqParser.cs`。

---

## Current Progress（目前進度）

測試狀態：**326 passed / 0 failed / 0 build warning**。工作區乾淨（HANDOFF.md 本身為 untracked，不需 commit）。

本 session 已完成（皆 TDD + 已驗證綠燈 + 已 commit）：

| 項目 | commit | 備註 |
|------|--------|------|
| 建立精簡版專案 `CLAUDE.md` | `4f137e660` | recipe + 雷點 + 檔案地圖 |
| `FOR JSON`（AUTO/PATH + ROOT/INCLUDE_NULL_VALUES/WITHOUT_ARRAY_WRAPPER） | `f7fd5606e` | 仿 FOR XML，單一類別 + mode enum |
| 視窗框架 `ROWS/RANGE BETWEEN` | `7ce9d347b` | 掛在聚合視窗函式路徑；RANK 路徑刻意不加（T-SQL 不合法） |
| 視窗框架限制註記（RANK / EXCLUDE） | `be54b890f` | `EXCLUDE` 標 `[N/A]`（SQL Server 不支援） |
| `WITHIN GROUP (ORDER BY)` | `d0ee71976` | 掛成 `SqlFunctionExpression.WithinGroup` |
| `GROUP BY ALL` | `63dfe1276` | `SqlGroupByClause.IsAll`；抽 `Parse_GroupBySimpleColumns` 共用 |
| `OPTION (query hint)` | `2d18e97ee` | 通用 hint 收集；**`OPTION` 已加入 ReservedWords** |
| `CHECK` 約束（欄位 + 資料表層級） | `34bf37c37` | 述詞重用 `Parse_WhereExpression`；break 條件加 `CHECK` |
| 欄位 `COLLATE` | `f1bc5c594` | `SqlColumnDefinition.Collation`，在 `ParseColumnConstraints` 解析 |
| 運算式 `COLLATE`（WHERE / ORDER BY） | `0dc90dda2` | `SqlCollateExpression`，後綴掛在 `Parse_Value_As_DataType`（OVER 之後、AS 之前） |
| `IN (subquery)` 回歸測試 | `cc4118297` | **既有功能**（`Parse_ConditionExpr` 的 IN 右側走 `parseTerm` 即可解析 `(SELECT…)`）；todo 原誤標未支援，補測試驗證後改 `[x]` |
| UNION 後 top-level `ORDER BY` | `8a45d8323` | 尾端 ORDER BY 改掛外層 `SelectStatement.OrderBy`；`ParseSelectStatement(asSetOperand)` 旗標讓 bare operand 不吃 ORDER BY，括號子查詢保留自身 ORDER BY |
| `TABLESAMPLE` | `517d79fc4` | `SqlTableSampleClause`（SYSTEM/PERCENT/ROWS/REPEATABLE）掛在 `SqlTableSource.TableSample`，於 alias 後、WITH 前解析；**`TABLESAMPLE` 已加入 ReservedWords** 否則被當別名吃掉 |
| `FOR XML RAW/EXPLICIT` | `145690c9c` | 單一 `SqlForXmlModeClause` + `SqlForXmlMode` enum（仿 FOR JSON）涵蓋 RAW（可選 `('elem')`）與 EXPLICIT，皆支援 ROOT directive；順手刪除空的 dead `ForXmlType.cs` |
| `INSERT` 解析（MVP） | _未 commit_ | additive 擴充 `SqlInsertStatement`（`ValuesRows`/`SourceSelect`/`IsDefaultValues`，builder 路徑不動）+ `ParseInsertStatement` dispatch（SELECT 後）；涵蓋 `INTO`/省略、`(cols)`/省略、單列+多列 VALUES、`INSERT...SELECT`、`DEFAULT VALUES`、VALUES 內運算式（函式/NULL）；Visitor 走訪 `ValuesRows`+`SourceSelect`；`INSERT` 由關鍵字起頭，**未加入 ReservedWords**（非語句尾關鍵字，不影響別名）。7 個新測試 `ParseInsertSqlTest.cs` |
| `INSERT` 第二階段（TOP/OUTPUT/hint/DEFAULT 值） | _未 commit_ | 同 `ParseInsertStatement` 擴充：`TOP (n)`（重用 `Parse_TopClause`）、`OUTPUT [INTO]`（新 `SqlOutputClause`，欄位走 `Parse_Column_Arithmetic`+AS-unwrap，不解析 bare alias 以避開 VALUES 被當別名）、目標 table hint（**抽共用 `Parse_WithTableHints`**，與 FROM hint 同源）、VALUES 列內 `DEFAULT`（新 `SqlDefaultValue`，僅 `Parse_InsertRowValue` 解析、不污染全域 `ParseValue`）。+5 測試 |
| `UPDATE` 解析 | _未 commit_ | additive 擴充 `SqlUpdateStatement`（`Top`/`Withs`/`SetClauses : List<SqlAssignExpr>`/`Output`/`FromSources`/`Where`，**重用既有 `SqlAssignExpr`**、不動 builder 的 `SetColumns`/`ToSql()`）；`ParseUpdateStatement`（INSERT 後 dispatch）：TOP/hint/SET 多指派/OUTPUT/FROM+JOIN/WHERE 全部重用既有 helper；SET 值走共用 `Parse_ValueOrDefault`（原 `Parse_InsertRowValue` 改名，支援 `= DEFAULT`）。8 測試 `ParseUpdateSqlTest.cs` |
| `DELETE` 解析 | _未 commit_ | 新增三處（`SqlDeleteStatement` + `SqlType.DeleteStatement` + `Visit_DeleteStatement`）；`ParseDeleteStatement`：**雙 FROM**（leading FROM 可省接 target、第二個 FROM 接 join）+ TOP/hint/OUTPUT/WHERE 全重用。6 測試 `ParseDeleteSqlTest.cs` |
| CTE 前綴接 DML | _未 commit_ | `ParseWithCteStatement` 改用新 `Parse_CteBodyStatement`（依序 try SELECT/INSERT/UPDATE/DELETE）；`SqlWithCte.Statement` 本就是 `ISqlExpression`、無需改 AST。`WITH cte AS (...) {INSERT\|UPDATE\|DELETE}` 三者皆通。3 測試 `ParseCteDmlTest.cs` |
| `MERGE` 解析（MVP） | _未 commit_ | 新 AST：`SqlMergeStatement` + `SqlMergeWhenClause`（+`MergeMatchType` enum）+ `ISqlMergeAction`／`SqlMergeUpdateAction`／`SqlMergeDeleteAction`／`SqlMergeInsertAction`（action 分離、避免單類別塞各欄位）；5 個 `SqlType` + 5 個 `Visit_Merge*`。`ParseMergeStatement`：target/source 用 `Parse_TableSourceWithHints`、ON/AND 用 `Parse_WhereExpression`、UPDATE action 用 `Parse_UpdateSetClause`、INSERT action 用 `Parse_ParenthesizedColumns`+`Parse_InsertValuesRow`；多 WHEN loop 收集。**`USING` 已加入 `ReservedWords`**（無 alias target 防呆）。4 測試 `ParseMergeSqlTest.cs` |
| DDL：`TRUNCATE TABLE` + `DROP` | _未 commit_ | `SqlTruncateTableStatement`；`SqlDropStatement` + `SqlDropObjectType` enum（Table/View/Procedure/Function/Index/Trigger/Schema/Database/Sequence/Type，含 PROC 別名）+ `IF EXISTS` + 多名稱（`ParseWithComma(Parse_SqlIdentifier)`）。dispatch 加在 MERGE 後。5 測試 `ParseDdlDropTruncateTest.cs`。**首個 DDL 寫入語句** |
| DDL：`ALTER TABLE`（MVP） | _未 commit_ | `SqlAlterTableStatement { TableName, Action : ISqlAlterTableAction }`；action 分離 5 類（AddColumns/AddConstraint/DropColumn/DropConstraint/AlterColumn）+ 6 `SqlType` + 6 Visit。**重用 `ParseColumnDefinition`（ADD/ALTER COLUMN）+ `ParseTableConstraint`（ADD CONSTRAINT，已含 CONSTRAINT name+PK/UNIQUE/FK/CHECK）**；ADD 分流靠 peek `CONSTRAINT`/`PRIMARY`/`UNIQUE`/`FOREIGN`/`CHECK`。6 測試 `ParseAlterTableSqlTest.cs` |
| DDL：`CREATE VIEW`（MVP） | _未 commit_ | `SqlCreateViewStatement { IsOrAlter, ViewName, ColumnNames, Query, WithCheckOption }`。`CREATE` 與 CREATE TABLE 共用 → 消費 CREATE 後若非 VIEW 會 **reset position** 再 return None；body 走 `Parse_CteBodyStatement`（支援 SELECT/CTE）；`WITH CHECK OPTION` 在 SELECT 之後解析。4 測試 `ParseCreateViewSqlTest.cs` |
| DDL：`CREATE INDEX`（MVP） | _未 commit_ | `SqlCreateIndexStatement { IsUnique, Clustered, IndexName, TableName, Columns, IncludeColumns, Where }`。`CREATE` 共用 → 消費後若非 INDEX **reset position**。重用 `ParseColumnsAscDesc`（key 欄位含 ASC/DESC）、`Parse_ParenthesizedColumns`（INCLUDE）、`Parse_WhereExpression`（filtered）。4 測試 `ParseCreateIndexSqlTest.cs` |
| DDL：`DROP INDEX ix ON table` | _未 commit_ | additive 擴充 `SqlDropStatement.OnTable`；`ParseDropStatement` 在 `ObjectType==Index` 時解析尾端 `ON table`。2 測試加入 `ParseDdlDropTruncateTest.cs` |
| DML 小單點收尾 | _未 commit_ | ① MERGE CTE 前綴（`Parse_CteBodyStatement` 加 MERGE）② MERGE `INSERT DEFAULT VALUES` action（補測試）③ UPDATE 複合指派 `+= -= *= /= %= &= \|= ^=`（additive `SqlAssignExpr.Operator` default `=` + `Parse_AssignOperator`）④ `INSERT ... EXEC proc [args]`（新 `SqlExecStatement` 掛 `SqlInsertStatement.ExecSource`；args 解析 guard `IsEnd()`/`;` 以容許無參數）。+5 測試 |
| 頂層 `EXEC proc [args]` | _未 commit_ | 重用 §上一列已建的 `Parse_ExecStatement`/`SqlExecStatement`，僅加 dispatch（排在 `ParseExecSpAddExtendedProperty` 之後，特定 sp 先比對）。3 測試 `ParseExecSqlTest.cs` |
| MERGE 第二階段 | _未 commit_ | additive 加 `SqlMergeStatement.Top`/`Output`/`Option`；`ParseMergeStatement` 加 TOP（MERGE 後、INTO 前，重用 `Parse_TopClause`）、OUTPUT + OPTION（WHEN loop 後，重用 `Parse_OutputClause`/`ParseOptionClause`）。target hint 已由 `Parse_TableSourceWithHints` 落在 `Target.Withs`（hint 須無 alias）。3 測試 |
| 控制流程（DECLARE/IF/WHILE/BEGIN…END） | _未 commit_ | 新 AST：`SqlDeclareStatement`(+`SqlVariableDeclaration`)、`SqlBlockStatement`、`SqlIfStatement`、`SqlWhileStatement` + 4 `SqlType` + 4 Visit。**body/分支用 `Parse()` 遞迴解析單一語句**（IF/WHILE 的 then/else/body 可為 `BEGIN…END`）；DECLARE 型別用 `ReadSqlIdentifier`+`Parse_DataSize`、值用 `ParseArithmeticExpr`；條件用 `Parse_WhereExpression`。`BEGIN` 對 `TRY`/`TRAN`/`TRANSACTION`/`CATCH` reset position 不攔截（留給未來）。8 測試（`ParseDeclareSqlTest.cs` 3 + `ParseControlFlowSqlTest.cs` 5） |
| `CREATE PROCEDURE`（MVP） | _未 commit_ | `SqlCreateProcedureStatement` + `SqlProcedureParameter`。`CREATE` 共用 → 消費後若非 `PROCEDURE`/`PROC` reset position。參數含 `(size)`/`= default`/`OUTPUT`，無 paren / 有 paren 皆可；`Parse_ProcedureParameter` 以「名稱須 `@` 開頭、否則 reset」區分無參數（讀到 `AS` 即停）。**body 直接重用 `Parse()`**（單一語句或 `BEGIN…END`）。4 測試 `ParseCreateProcedureSqlTest.cs` |
| `RETURN` + `CREATE FUNCTION`（MVP） | _未 commit_ | `SqlReturnStatement`（值走 `ParseArithmeticExpr`，bare RETURN 在 `END`/`;`/EOF 前不取值）；`SqlCreateFunctionStatement`（scalar `RETURNS type[(size)]` + inline TVF `RETURNS TABLE AS RETURN (select)`；params 重用 `Parse_ProcedureParameter`、body 重用 `Parse()`）。`CREATE` 共用 → 非 FUNCTION reset position。5 測試（`ParseReturnSqlTest.cs` 2 + `ParseCreateFunctionSqlTest.cs` 3） |
| `BEGIN TRY…CATCH` + `TRANSACTION` | _未 commit_ | 新 AST：`SqlTryCatchStatement`、`SqlTransactionStatement`(+`SqlTransactionAction` enum) + 2 `SqlType` + 2 Visit。**抽共用 `ParseStatementsUntil(params endKeywords)`** 解析「迴圈 `Parse()` 到指定結尾關鍵字序列」，並 **refactor `ParseBlockStatement` 復用**（行為不變、原 ControlFlow 測試保護）。`ParseTryCatchStatement`：BEGIN 後非 TRY 即 reset；try/catch body 各 `ParseStatementsUntil("END","TRY")`/`("END","CATCH")`。`ParseTransactionStatement`：`BEGIN|SAVE TRAN[SACTION]`、`COMMIT|ROLLBACK [TRAN\|TRANSACTION\|WORK]`，選擇性交易名稱用 `TransactionNameBoundaryKeywords` stop-set 擋後續語句關鍵字（避免吃掉 `COMMIT`/`END` 等）。dispatch 排在 `ParseBlockStatement` 前。8 測試（`ParseTryCatchSqlTest.cs` 2 + `ParseTransactionSqlTest.cs` 6） |
| 具名 `WINDOW` 子句（MVP） | `84d24ff12` | `SqlWindowClause`/`SqlWindowDefinition` 掛 `SelectStatement.Window`（HAVING 後、ORDER BY 前）+ `func() OVER name`（`SqlOverWindowName`）；**`WINDOW` 已加入 ReservedWords**；改 `ParseOverOrderByClause` 在無 `(` 時 reset 位置，讓新的 bare `OVER name` 能接在後面試；行內延伸/互參照延後（見 todo §4） |

---

## What Worked（沿用上一個 session 的 recipe，已收進 `CLAUDE.md`）

每加一個語法功能：型別表面先行（AST enum/屬性/類別 + `SqlType` + `SqlVisitor` 三處同步）→ 寫測試確認紅 → 實作 parser 確認綠 → 跑完整套件零回歸 + 0 警告 → 打勾 `plans/todo-tsql.md`。詳見 `CLAUDE.md`。

本 session 額外驗證有效的做法：
- **誠實標記限制**：對「非 T-SQL」語法（如 frame 的 `EXCLUDE`）用 `[N/A]` 而非硬做；對「只做了一部分」用 `[~]` 並註記（如 `OPTION` 不驗證 hint 合法性、`COLLATE` 只做欄位定義層級）。
- **重用優先**：CHECK 述詞直接用 `Parse_WhereExpression`、WITHIN GROUP 內部 ORDER BY 用 `ParseOrderByClause`，不重造。

---

## Gotchas（本 session 踩到 / 確認的，補充 `CLAUDE.md` 已記的）

1. **ReservedWords**：`OPTION` 這次必須加入 `ReservedWords`（line ~14），否則 `FROM t OPTION (...)` 會把 `OPTION` 當 table 別名吃掉。加新的「語句尾關鍵字」時都要評估這點。
2. **多處 OVER 解析**：`ParseRankClause`、`ParseOverOrderByClause`、`ParseOverPartitionByClause` 三條路徑各自解析 OVER；視窗框架只加在後兩條（泛用值 + OVER），RANK 路徑沒加（T-SQL 排名函式不允許 frame）。
3. **CREATE TABLE 欄位 vs 資料表約束分界**：`ParseCreateTableColumns`（line ~650）有個 break 條件用 `PeekKeywords(...)` 判斷「這列不是欄位、是資料表約束」；新增資料表層級約束關鍵字（這次加了 `CHECK`）要同步加進去。
4. **`required` 慣例**：新 AST 的必填參考型別屬性用 `required`（如 `SqlConstraintCheck.Predicate`），與 `SqlConditionExpression` 一致。

---

## Next Steps（下一步，依優先序）

清單在 `plans/todo-tsql.md`，目前剩餘：

1. 🟢 **DDL 續攻**（DROP/TRUNCATE/ALTER TABLE/CREATE VIEW/CREATE INDEX/DROP INDEX ON / CREATE PROCEDURE/FUNCTION 已完成）：`CREATE TRIGGER`（含 FOR/AFTER/INSTEAD OF + 語句 body）、multi-statement TVF `RETURNS @t TABLE(...)`、ALTER TABLE 第二階段（見 §1.5）。
2. 🟢 **控制流程續攻**（DECLARE/IF/WHILE/BEGIN…END/RETURN/TRY…CATCH/TRANSACTION 已完成）：`BREAK`/`CONTINUE`、`PRINT`/`THROW`/`RAISERROR`、`DECLARE @t TABLE`、`BEGIN DISTRIBUTED TRANSACTION`。
3. 🟢 **DML 細項剩餘**（皆小）：`EXEC ('dynamic sql')`、EXEC 具名參數 `@p = val`、MERGE `OUTPUT $action`。

**立即動作建議**：DML + DDL（含 `CREATE PROCEDURE/FUNCTION`）+ 控制流程（含 RETURN/TRY…CATCH/TRANSACTION）皆已備齊。下一步可挑 **`PRINT`/`THROW`/`RAISERROR`**（簡單，常與 CATCH 搭配）、**`BREAK`/`CONTINUE`**（WHILE 配套，簡單），或 **`CREATE TRIGGER`** / multi-statement TVF（較大）。

> ⚠️ **commit 雷點**：git repo root 是上層的 `Samples/`，不是 `T1.SqlSharp/`。**絕對不要用 `git add -A` / `git add .`**，會把 repo 根一堆無關 untracked（`openSource/` 內嵌 git repo、`gsoft/`、大型二進位）和本檔（`HANDOFF.md`，刻意 untracked）一起 commit。一律用「明確列出檔案路徑」的 `git add <path...>`。

---

## INSERT 解析 — 範圍與設計建議（下一個 session 的藍圖）

> 結論先說：**現有 `SqlInsertStatement` 是 builder 專用形狀，不能直接拿來當 parser 的輸出**。要嘛additively 擴充、要嘛另立解析型別。先讀完本節再動手，省得走回頭路。

### A. 為何不能直接複用現有 AST（核心約束）

`T1.SqlSharp/Expressions/SqlInsertStatement.cs` 目前是這形狀：

```
TableName : string
Columns   : List<string>          // 只有欄名字串
ToSql()                           // 固定輸出 (...) VALUES (@p0, @p1, ...)
```

它被這些地方消費（改形狀會破壞它們，動前先看）：

- `Helper/SqlInsertExpressionBuilder.cs`（LINQ-style builder，`Into(dbSet).Build()`）
- `T1.SqlSharpTests/SqlInsertExpressionBuilderTest.cs`、`SqlVisitorTest.cs:305`
- `SqlUpdateStatement` 同理：`SetColumns` 帶 `ParameterName`（`@p`），被 `SqlUpdateExpressionBuilder` + `SqlUpdateExpressionBuilderTest` 消費。

問題：parser 要表達的是「`VALUES (1, 'a', GETDATE())`、多列、任意運算式」或「`INSERT ... SELECT ...`」，現有 `Columns: List<string>` + 參數化 `ToSql()` **承載不了**。

### B. AST 設計：建議「additive 擴充」而非另立新類別

兩個選項，建議選 1：

1. **（建議）additive 擴充 `SqlInsertStatement`**：保留 `TableName` / `Columns`（builder 仍用），**新增 nullable 解析欄位**，parser 走新欄位、builder 走舊欄位，互不干擾：
   - `ValuesRows : List<List<ISqlExpression>>`（`= []`；每列一組運算式，支援多列 VALUES）
   - `SourceSelect : SelectStatement?`（`INSERT ... SELECT`）
   - `IsDefaultValues : bool`（`INSERT ... DEFAULT VALUES`）
   - `OutputClause : ...?`（`OUTPUT inserted.*`，可第二階段再做）
   - 風險：現有 `ToSql()` 是參數化輸出，round-trip 解析後的 ToSql 會對不上。**第一階段不要動 `ToSql()`**（builder 測試靠它），解析後的 ToSql 列為 known limitation 或另開 method。
   - 符合 CLAUDE.md「重用優先 / 不複製成多類別」。
2. 另立 `SqlInsertParsedStatement`：乾淨但與 builder 型別重複，違反「不要複製成多個類別」，**不建議**。

> 三處同步照 recipe：`SqlType` 已有 `InsertStatement`（不用加 enum）；`SqlVisitor.Visit_InsertStatement` 已存在但只 `AddSqlExpression`——擴充後要**走訪新子節點**（`ValuesRows` 內運算式、`SourceSelect`），否則 §雷點 3 那種「子查詢沒被走訪」會重演。

### C. Parser 整合點

1. **頂層 dispatch**：`SqlParser.Parse()`（line 65-93）加一條 `Try(ParseInsertStatement, out var insert)`。位置放在 SELECT 之後、SET 之前即可（INSERT 由 `INSERT` 關鍵字開頭，不會與 SELECT 衝突）。
2. **`ParseInsertStatement` 骨架**（全部重用既有 helper，不要重造）：
   - `INSERT` + optional `INTO` → 用 `TryKeyword`
   - target table → **`ParseTableName`**（line ~500，支援 schema.table / #temp）
   - optional 欄位清單 `(c1, c2, ...)` → **`Parse_ParenthesizedColumns`**（line 308）
   - 分支：
     - `VALUES` → 解析 1..N 列，每列 `(` 運算式逗號清單 `)`，運算式用 **`ParseArithmeticExpr`** / `Parse_Value_As_DataType`
     - `SELECT` / `WITH` → **`ParseSelectStatement()`**（line 957）掛 `SourceSelect`
     - `DEFAULT VALUES` → 設 `IsDefaultValues`
3. **回傳 `ParseResult<SqlInsertStatement>`**，沿用 `CreateParseResult` / `CreateParseError` 慣例。

### D. 範圍切分（建議分多個 commit，逐步 TDD）

第一階段（MVP，✅ 本 session 已全部綠燈，測試在 `ParseInsertSqlTest.cs`）：
- [x] `INSERT INTO t (a, b) VALUES (1, 'x')` — 單列
- [x] 多列 `VALUES (..), (..), (..)`
- [x] 省略欄位清單 `INSERT INTO t VALUES (...)`
- [x] 省略 `INTO`（`INSERT t ...`，T-SQL 合法）
- [x] `INSERT INTO t (cols) SELECT ...`
- [x] `INSERT INTO t DEFAULT VALUES`
- [x] VALUES 內運算式（測了 `GETDATE()`、`NULL`；`a + 1`/negative/CASE 走同一 `ParseArithmeticExpr`，未個別補測）

第二階段（✅ 大部分本 session 已綠燈）：
- [x] `OUTPUT col [AS alias] [INTO target [(cols)]]` 子句
- [x] `INSERT TOP (n) [PERCENT] ...`
- [x] 目標 table hint `WITH (TABLOCK)`
- [x] `VALUES` 列內 `DEFAULT` 關鍵字當值
- [ ] `INSERT ... EXEC proc` / `EXEC ('sql')`（rowset 來源，仍待做）
- [ ] CTE 前綴 `WITH cte AS (...) INSERT ...`（需擴充 `SqlWithCte.Statement` 接受 INSERT；目前 `ParseWithCteStatement` 寫死接 `ParseSelectStatement`）

UPDATE / DELETE（INSERT 完再開）：
- UPDATE：target + `SET col = expr [, ...]`（expr 用 `ParseArithmeticExpr`，**非**參數化）+ optional `FROM` + `WHERE`（重用 `Parse_WhereExpression`）。同樣要 additive 擴充 `SqlUpdateStatement`（現有 `SetColumns.ParameterName` 是 builder 用，解析要存 `ISqlExpression` 值）。
- DELETE：`DELETE [FROM] t [FROM ...] [WHERE ...]`，無現成 AST，需新增 `SqlDeleteStatement` + enum + visitor 三處。

### E. 雷點預判（動手前先想）

1. **ReservedWords**：實測 `INSERT`/`VALUES`/`OUTPUT`/`DEFAULT` 都**不需**加入 `ReservedWords`——它們都在 `ParseInsertStatement` 內以位置順序明確消費（非語句尾、非別名位置），故未動 `ReservedWords`。但 OUTPUT 欄位是雷點：因 `Parse_SelectColumns` 只排除 `FROM`/`INTO` 當別名，故 OUTPUT 改用 `Parse_Column_Arithmetic`（不吃 bare alias）來避免把後面的 `VALUES`/`SELECT` 當成欄位別名。`UPDATE`/`DELETE` 接手時再各自評估。
2. **`VALUES` 與「衍生表 VALUES constructor」**：T-SQL 也有 `FROM (VALUES (1),(2)) AS t(x)`，若之後要支援，VALUES 列解析邏輯可抽共用方法，現在先不抽、先讓 INSERT 自用，避免過度設計。
3. **Visitor 走訪**：擴充後務必讓 `Visit_InsertStatement` 走訪 `ValuesRows` 與 `SourceSelect`（見 B 節末）。
4. **builder 測試不可破**：`SqlInsertExpressionBuilderTest` / `SqlUpdateExpressionBuilderTest` / `SqlVisitorTest` 必須全綠——additive 設計就是為了保這個。改完跑全套件確認 223+ 仍全綠。

### F. TDD 第一步（紅燈起手式）

在 `T1.SqlSharpTests/` 新增 `ParseInsertSqlTest.cs`，第一個測試最小：

```
"INSERT INTO Users (Id, Name) VALUES (1, 'Alice')".ParseSql()
  → SqlInsertStatement { TableName = "Users",
        Columns = [...] 或 新的欄位表示,
        ValuesRows = [[ SqlValue{IntValue,1}, SqlValue{"'Alice'"} ]] }
```

跑 → 確認紅（且紅因為 parser dispatch 不認 INSERT，而非編譯錯）→ 再實作 `ParseInsertStatement`。

---

## 常用指令

```bash
dotnet test                                              # 全套件（自動 build）
dotnet test --filter "FullyQualifiedName~ForJson"        # 只跑某組
dotnet build T1.SqlSharp/T1.SqlSharp.csproj --no-incremental 2>&1 | grep -i warning  # 確認 0 警告
```

## 關鍵檔案

- 開發規則 / 雷點 / 檔案地圖：**`CLAUDE.md`**（專案根，先讀這個）
- Parser：`T1.SqlSharp/ParserLit/SqlParser.cs`（主）、`LinqParser.cs`
- AST：`T1.SqlSharp/Expressions/*.cs`（`SqlType.cs` enum、`SqlVisitor.cs` 走訪）
- 測試：`T1.SqlSharpTests/*.cs`（helper：`TestHelper.cs`）
- 待辦清單：`plans/todo-tsql.md`
