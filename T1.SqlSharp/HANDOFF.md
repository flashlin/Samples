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

測試狀態：**461 passed / 0 failed / 0 build warning**。工作區乾淨（HANDOFF.md 本身為 untracked，不需 commit）。

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
| `PRINT`/`THROW`/`RAISERROR` + `BREAK`/`CONTINUE` | _未 commit_ | 新 AST：`SqlPrintStatement`、`SqlThrowStatement`、`SqlRaiseErrorStatement`、`SqlLoopControlStatement`(+`SqlLoopControlAction` enum) + 4 `SqlType` + 4 Visit。`PRINT` 值走 `ParseArithmeticExpr`；`THROW` bare 或三參數（`ParseWithComma`→ErrorNumber/Message/State）；`RAISERROR (msg,sev,state[,args]) [WITH opt,...]`（前三必填、其餘入 `Arguments`、WITH 選項 `ReadSqlIdentifier` loop 入 `Options`）；`BREAK`/`CONTINUE` 關鍵字語句共用單類別 enum。dispatch 排在 `ParseReturnStatement` 後。**回歸修正**：`ExcludeNonSelectStatementTest.ExtractKnownStatements` 原用 `print '123'` 當被略過的未知語句，PRINT 已支援故改用 `use mydb`。9 新測試（`ParsePrintThrowRaiseErrorSqlTest.cs` 6 + `ParseLoopControlSqlTest.cs` 3） |
| `CREATE TRIGGER` + `USE`/`GO` | _未 commit_ | 新 AST：`SqlCreateTriggerStatement`(+`SqlTriggerTiming`/`SqlTriggerEvent` enum)、`SqlUseStatement`、`SqlGoStatement` + 3 `SqlType` + 3 Visit。`CREATE TRIGGER name ON target {FOR\|AFTER\|INSTEAD OF} {INSERT\|UPDATE\|DELETE}[, ...] AS <body>`：`CREATE` 共用 → 非 TRIGGER reset position；timing 用 `ParseTriggerTiming`（INSTEAD OF 先比，雙關鍵字）、events 用 `ParseTriggerEvents`（逗號 loop）、body 重用 `Parse()`；target/name 用 `Parse_SqlIdentifier`（含 `dbo.x` 點名）。`USE db`、`GO [count]`（count 經 `ParseArithmeticExpr` 取 IntValue、非整數則 reset）排在 loop-control 後。**回歸修正（連鎖）**：上一輪改用的 `use mydb` 現已被 USE 支援，故 `ExtractKnownStatements` 再改用不衝突任何 dispatch 關鍵字的 `dbcc checkdb`。6 新測試（`ParseCreateTriggerSqlTest.cs` 3 + `ParseUseGoSqlTest.cs` 3） |
| multi-statement TVF + `GRANT`/`REVOKE`/`DENY` | _未 commit_ | ① **additive 擴充 `SqlCreateFunctionStatement`**（`ReturnTableVariable`/`ReturnTableColumns`，不動既有 scalar/inline 路徑與 ToSql 形狀）；**refactor `ParseCreateFunctionStatement` 抽 `ParseFunctionReturnClause`**（return clause 三分支：`@var TABLE (cols)` multi-statement TVF / `TABLE` inline / scalar `type[(size)]`），TVF 欄位重用 `ParseColumnDefinition`；Visitor 補走訪 `ReturnTableColumns`。② 新 AST：`SqlPermissionStatement` + `SqlPermissionAction` enum（Grant/Revoke/Deny）+ 1 `SqlType` + 1 Visit。`{GRANT\|REVOKE\|DENY} perm[, ...] [ON securable] {TO\|FROM} principal[, ...] [WITH GRANT OPTION] [CASCADE]`；perm/principal 用共用 `ReadCommaSeparatedIdentifiers`、securable 用 `Parse_SqlIdentifier`、action 用 `TryParsePermissionAction`。dispatch 排在 GO 後。6 新測試（`ParseCreateFunctionSqlTest.cs` +1 = 4 + `ParsePermissionSqlTest.cs` 5） |
| `CREATE SCHEMA`/`DATABASE` + ALTER TABLE 第二階段 | _未 commit_ | ① 新 AST：`SqlCreateSchemaStatement`（`SCHEMA name [AUTHORIZATION owner]`）、`SqlCreateDatabaseStatement`（`DATABASE name`）+ 2 `SqlType` + 2 Visit。**共用一個 `ParseCreateSchemaOrDatabaseStatement()`（回 `ParseResult<ISqlExpression>`）消費 `CREATE` 後分流 SCHEMA/DATABASE，非兩者則 reset position**；dispatch 排在 CREATE TRIGGER 後（dispatch 直接 `return` 該 `ParseResult`，非 `.Result`）。② ALTER TABLE 第二階段：新 action `SqlAlterTableToggleTrigger`（`{ENABLE\|DISABLE} TRIGGER {ALL\|names}`）、`SqlAlterTableCheckConstraint`（`{CHECK\|NOCHECK} CONSTRAINT {ALL\|names}`）+ 2 `SqlType` + 2 Visit；**additive `SqlAlterTableAddConstraint.WithCheck : bool?`**（`WITH CHECK`/`WITH NOCHECK` 前綴在 `Parse_AlterTableAction` 開頭解析、threa 進 `Parse_AlterTableAddAction(bool?)`）。8 新測試（`ParseCreateSchemaDatabaseSqlTest.cs` 3 + `ParseAlterTableSqlTest.cs` +5） |
| 🎯 拉高完成度批次（goal：20/20 完成） | _未 commit_ | 依使用頻率挑高頻項，全 TDD（382→419，+37 測試，0 警告，dispatch ~48 種）。**1–10**：① `SET TRANSACTION ISOLATION LEVEL` ② `BEGIN DISTRIBUTED TRANSACTION`（`IsDistributed`）③ `GOTO`/`label:`（label dispatch 置末）④ `CHECKPOINT`/`RECONFIGURE`/`REVERT`/`SHUTDOWN`（`SqlKeywordStatement`）⑤ `CREATE SEQUENCE`（clause loop）⑥ `NEXT VALUE FOR`（`SqlNextValueForExpr`，hook `ParseArithmetic_Primary`）⑦ `CREATE TYPE`（FROM/AS TABLE）⑧ `CREATE/DROP SYNONYM` ⑨ `ALTER INDEX`（REBUILD/REORGANIZE/DISABLE）⑩ `FROM (VALUES…) AS t(c)`（`SqlValuesTableSource`）。**11–20**：⑪ `OPENJSON`/`OPENQUERY` 來源（既有 TVF 路徑，補測試）⑫ `AT TIME ZONE`（`SqlAtTimeZoneExpr` postfix）⑬ `CONTAINS`/`FREETEXT`（既有泛用函式，補測試）⑭ `ALTER DATABASE SET` ⑮ `ALTER SCHEMA TRANSFER` ⑯ `CREATE LOGIN/USER/ROLE`（`SqlCreatePrincipalStatement`）⑰ `CREATE/UPDATE STATISTICS`（`SqlStatisticsStatement`）⑱ `CREATE INDEX WITH(options)` ⑲ `DECLARE CURSOR` 進階選項（`CursorOptions`）⑳ `GRANT ON class::securable`（`SecurableClass`）。每項 1 檔測試 |
| 🎯 拉高完成度批次二（goal：20/20 完成，419→461，+42 測試） | _未 commit_ | 依使用頻率挑高頻項，全 TDD、零回歸、0 警告。**1–5**：① `GRANT` 多字權限（`VIEW DEFINITION`/`CREATE TABLE`，新 `ReadPermissionNames` 逐字收集至 `,`/`ON`/`TO`/`FROM`）② 前導 `;THROW`（`Parse()` 開頭 `while(TryMatch(";"))` 通用語句分隔）③ `ALTER ROLE {ADD\|DROP} MEMBER`（`SqlAlterRoleStatement`）④ `ALTER {LOGIN\|USER}`（`SqlAlterPrincipalStatement`，ENABLE/DISABLE/`WITH K=V` 走新 `ReadAssignmentOptionList`）⑤ `DROP {LOGIN\|USER\|ROLE}`（擴 `SqlDropObjectType`）。**6–10**：⑥ `DBCC cmd [(args)] [WITH opt]`（`SqlDbccStatement`）⑦ `BULK INSERT t FROM 'f' [WITH (opt)]`（`SqlBulkInsertStatement`）⑧ `ALTER SEQUENCE`（`SqlAlterSequenceStatement`，RESTART/INCREMENT）⑨ `CREATE DATABASE` ON/LOG ON/COLLATE（additive `SqlCreateDatabaseStatement`，新 `ReadFileSpecList`）⑩ `FOR SYSTEM_TIME`（`SqlTableSource.ForSystemTime`，`Parse_ForSystemTime` 於 alias 前；bound 用 `ParseArithmetic_Primary` 避免吃 `AND`）。**11–15**：⑪ `OPENJSON(...) WITH (col type ['path'] [AS JSON])`（`SqlFuncTableSource.JsonSchemaColumns`，`ReadOpenJsonSchema`）⑫⑬ `BACKUP`/`RESTORE`（單一 `SqlBackupRestoreStatement` + `IsBackup`）⑭ `KILL [WITH STATUSONLY]`（`SqlKillStatement`）⑮ `REVOKE GRANT OPTION FOR`/`AS grantor`（additive `SqlPermissionStatement`）。**16–20**：⑯ ODBC 跳脫 `{ fn\|d\|t\|ts ... }`（`SqlOdbcEscapeExpr`，hook `ParseArithmetic_Primary` 偵測 `{`）⑰ `$PARTITION.fn(...)`（**既有泛用函式路徑**，FunctionName=`$PARTITION.fn`，僅補回歸測試）⑱ `CREATE FULLTEXT INDEX`（`SqlCreateFulltextIndexStatement`）⑲ `CREATE PARTITION FUNCTION/SCHEME`（兩類別）⑳ `CREATE XML SCHEMA COLLECTION`（`SqlCreateXmlSchemaCollectionStatement`）。**回歸修正**：`ExtractKnownStatements` 原用 `dbcc checkdb` 當被略過的未知語句，DBCC 已支援故改用 `dump database mydb`（DUMP 永不在範圍）。dispatch 頂層語句 ~48→~59 |
| `WAITFOR` + SET 取值型 | _未 commit_ | ① `WAITFOR {DELAY\|TIME} 'time'`：新 `SqlWaitForStatement` + `SqlWaitForKind` enum + 1 `SqlType` + 1 Visit；time 走 `ParseArithmeticExpr`。② SET 取值型：**refactor `SqlSetOptionStatement` 的 `bool IsOn` → `string Value`**（統一 ON/OFF 與取值；既有 3 個 ON/OFF 測試更新為 `Value="ON"/"OFF"` 作重構保護）；`ParseSetOptionStatement` 改用 `TryMatchOnOff`（先比 ON/OFF，含 reserved word），否則 `ParseArithmeticExpr().ToSql()` 取值，若其後接 ON/OFF 則首值存 `Target`（IDENTITY_INSERT）。涵蓋 `SET ROWCOUNT 100`/`SET DATEFORMAT mdy`。4 新測試（`ParseWaitForSqlTest.cs` 2 + `ParseSetOptionSqlTest.cs` +2 取值型） |
| `SET <option> {ON\|OFF}` session 選項 | _未 commit_ | **補 todo 缺口（原本會報錯）**：`SET NOCOUNT ON` 等無 `=` 的 session 選項，舊 `ParseSetValueStatement` 會 `Expected =` 報錯。新 `SqlSetOptionStatement`（Option/Target/IsOn）+ 1 `SqlType` + 1 Visit；新 `ParseSetOptionStatement` 排在 `ParseSetValueStatement` **前**：讀 option 名後若 peek `=` 或無 ON/OFF 則 **reset position 落回變數賦值**（additive，`SET @x = 1` 不受影響）；非 ON/OFF 時讀一個 target 識別字（支援 `SET IDENTITY_INSERT table ON`）。4 新測試（`ParseSetOptionSqlTest.cs`，含 `SET @x = 1` 回歸守護） |
| 全域變數 `@@x`（回歸驗證，既有功能） | _未 commit_ | TDD 探查證實 **既有 field reader 已整串擷取 `@@FETCH_STATUS`/`@@ROWCOUNT` 為 `SqlFieldExpr`**（如同 `$action`），可用於 WHILE 條件/SELECT 欄位 → **無需改 parser**，僅補回歸測試。2 新測試（`ParseGlobalVariableSqlTest.cs`：`SELECT @@ROWCOUNT`、`WHILE @@FETCH_STATUS = 0 ...`） |
| 游標操作 `OPEN`/`CLOSE`/`DEALLOCATE`/`FETCH` | _未 commit_ | 新 AST：`SqlCursorOperationStatement`（+`SqlCursorOperation` enum Open/Close/Deallocate，**單類別 enum** 因僅差 mode）、`SqlFetchStatement`（Direction/RowCount/CursorName/IntoVariables）+ 2 `SqlType` + 2 Visit。`ParseCursorOperationStatement`：OPEN/CLOSE/DEALLOCATE + cursor 名（`Parse_SqlIdentifier`，含 `@var`）。`ParseFetchStatement`：`FETCH [NEXT\|PRIOR\|FIRST\|LAST\|ABSOLUTE n\|RELATIVE n] [FROM] cur [INTO @v[, ...]]`（方向用 `ParseFetchDirection`，ABSOLUTE/RELATIVE 取 `ParseArithmeticExpr` 入 RowCount；FROM 可省；INTO 用 `ReadCommaSeparatedIdentifiers`）。dispatch 排在 loop-control 後。6 新測試（`ParseCursorOperationSqlTest.cs`） |
| 多字 WITH 選項 + `DECLARE {@c\|name} CURSOR` | _未 commit_ | ① **`Parse_WithOptionList` 改為逐元素 `Parse_WithOption()`**：辨識多字選項 `EXECUTE AS <principal>`、`RETURNS NULL ON NULL INPUT`、`CALLED ON NULL INPUT`（`TryKeywords` 比對固定字序），其餘走單字 `ReadSqlIdentifier`。**單字選項輸出不變**（向後相容）。② DECLARE 游標：`Parse_VariableDeclaration` 在 dataType=="CURSOR" 時走新 `Parse_CursorDeclaration(name)`（additive `SqlVariableDeclaration.IsCursor`/`CursorSource`）；`{@c\|name} CURSOR [FOR <select>]`，FOR 後重用 `ParseSelectStatement()`。4 新測試（`ParseDdlWithOptionsSqlTest.cs` +2、`ParseDeclareSqlTest.cs` +2） |
| DDL `WITH` 選項 + ALTER TABLE ADD 混合欄位+約束 | _未 commit_ | ① 四個 CREATE/ALTER 語句（VIEW/PROC/FUNCTION/TRIGGER）加 **additive `List<string> Options`**，於正確位置（VIEW/PROC/FUNCTION 在 AS 前、TRIGGER 在 ON 後 timing 前）以**共用 `Parse_WithOptionList()`**（`WITH` + `ReadCommaSeparatedIdentifiers`）解析；ToSql 補 `WITH ...`。**VIEW 的 `WITH CHECK OPTION`（query 後）與 pre-AS `WITH opt` 不衝突**。MVP 僅單字選項（SCHEMABINDING/ENCRYPTION/RECOMPILE）。② ALTER TABLE ADD：`Parse_AlterTableAddAction` 改為逗號 loop 收集欄位/約束（`IsTableConstraintStart` peek 分流），**向後相容形狀**：純欄位→`SqlAlterTableAddColumns`、單一純約束→`SqlAlterTableAddConstraint`（保留 `WithCheck`）、混合/多約束→新 `SqlAlterTableAddElements`（+1 `SqlType`+1 Visit）。6 新測試（`ParseDdlWithOptionsSqlTest.cs` 5 + `ParseAlterTableSqlTest.cs` +1） |
| `ALTER VIEW`/`PROCEDURE`/`FUNCTION`/`TRIGGER` | _未 commit_ | ALTER 版 body 與 CREATE 完全相同 → **DRY：讓四個 CREATE parser 的開頭共用 `TryDefinitionLead(out startSpan, out isAlter, out isOrAlter)`**（消費 `CREATE [OR ALTER]` 或 `ALTER`），各 parser 對應 AST 加 **additive `bool IsAlter`**（三態：CREATE / CREATE OR ALTER / ALTER）；ToSql 前綴抽共用 `DefinitionLead.ToSql(isAlter, isOrAlter, keyword)`。**CREATE INDEX/TABLE 不接受 ALTER**（語法不同），`ALTER TABLE` 仍由其專屬 parser 處理（CREATE 系列遇 ALTER 但物件關鍵字不符會 reset，dispatch 順序安全）。重構由既有 `CREATE OR ALTER` 測試保護。4 新測試（`ParseAlterObjectSqlTest.cs`） |
| DML 細項收尾（EXEC 動態 SQL/具名參數、MERGE `OUTPUT $action`、`DECLARE @t TABLE`） | _未 commit_ | ① **抽共用 `Parse_ParenthesizedColumnDefinitions()`**（`( col defs )`），**refactor multi-statement TVF（`ParseFunctionReturnClause`）復用**，並供 `DECLARE @t TABLE` 使用。② EXEC：`EXEC ('sql' \| @sql)` 動態 SQL（additive `SqlExecStatement.DynamicSql`，peek `(` 分流 `Parse_ExecDynamicSql`）；具名參數 `@p = val [OUTPUT]`（新 `SqlExecArgument` + `SqlType.ExecArgument` + Visit，`Parse_ExecArgument` peek `@name =` 否則 reset 回 positional 裸運算式——**additive，既有 positional 測試不動**）。③ `DECLARE @t TABLE (cols)`（additive `SqlVariableDeclaration.IsTable`/`TableColumns`，dataType=="TABLE" 且 peek `(` 時走表變數路徑）。④ **MERGE `OUTPUT $action` 經驗證既有 field reader 已支援**（`$action` 解析為 `SqlFieldExpr`，先寫測試確認綠、無需改 parser）。6 新測試（`ParseExecSqlTest.cs` +4、`ParseDeclareSqlTest.cs` +1、`ParseMergeSqlTest.cs` +1） |
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

清單在 `plans/todo-tsql.md`。T-SQL 語句層級已近乎全覆蓋（dispatch ~59 種頂層語句），目前剩餘多為單一語句的進階選項微調：

1. 唯一 `[ ]`：`ALTER COLUMN` 的 `ADD/DROP` 子選項（罕見）。
2. `[~]` 進階選項缺口（皆非阻斷常用語法）：`EXEC ... AT linked_server`、`CREATE DATABASE` 帶單位 `SIZE = 10MB`/FILEGROUP、`CREATE SEQUENCE` 儲存 MIN/MAX/CYCLE/CACHE 值、`ALTER ROLE ... WITH NAME =`、欄位層級 `GRANT SELECT (col)`、`OPENJSON` 欄位 `LANGUAGE`/FULLTEXT 欄位 `LANGUAGE`、具名 WINDOW 行內延伸、表變數內 table 約束。
3. 建議下一步：與使用者確認是否要收尾這些 `[~]` 邊角，或轉向其他模組（如 `LinqParser` 擴充、ToSql round-trip 強化）。

**完成度**：依語法項目加權（`[x]`=1、`[~]`=0.5，共 194 項可追蹤、另 1 項 `[N/A]` 不計）≈ **92.8%**（180/194）；若「部分支援」算可用則 ≈ **99.5%**（193/194）。

**立即動作建議**：T-SQL 幾乎全覆蓋，**僅剩 1 個 `[ ]`**：`ALTER COLUMN` 的 `ADD/DROP` 子選項（罕見）。其餘 `[~]`（26 項）皆為單一語句的進階選項缺口，例如：`EXEC ... AT linked_server`、`CREATE DATABASE` 帶單位的 `SIZE = 10MB`/FILEGROUP、`CREATE SEQUENCE` 儲存 MIN/MAX/CYCLE/CACHE 值、`ALTER ROLE ... WITH NAME =` 改名、欄位層級 `GRANT SELECT (col)`、具名 WINDOW 行內延伸、表變數內 table 約束。建議與使用者確認下一個重點再續。

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
