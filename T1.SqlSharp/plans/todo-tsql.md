# T1.SqlSharp — T-SQL 語法支援清單

> 用途：追蹤 parser 目前支援哪些 T-SQL 語法，方便維護與規劃。
> 圖例：`[x]` 已支援、`[ ]` 未支援、`[~]` 部分支援、`[N/A]` 不適用 T-SQL（不實作）。
> 最後驗證：2026-06-21（依 `T1.SqlSharp/ParserLit/SqlParser.cs`、`LinqParser.cs` 與測試實際比對；382 測試全綠）。
> 入口：`SqlParser.Parse()` dispatch ~36 種頂層語句（WITH CTE / CREATE TABLE|VIEW|INDEX|PROCEDURE|FUNCTION|TRIGGER|SCHEMA|DATABASE / SELECT / INSERT / UPDATE / DELETE / MERGE / TRUNCATE / DROP / ALTER TABLE / EXEC sp_addextendedproperty / EXEC proc / DECLARE / BEGIN TRY…CATCH / TRANSACTION（BEGIN|COMMIT|ROLLBACK|SAVE）/ BEGIN…END / IF / WHILE / RETURN / PRINT / THROW / RAISERROR / BREAK / CONTINUE / OPEN|CLOSE|DEALLOCATE / FETCH / WAITFOR / USE / GO / GRANT|REVOKE|DENY / SET（變數賦值 + session 選項 ON/OFF/取值））。

---

## 1. 頂層語句 (Top-level statements)

- [x] `SELECT`
- [x] `WITH cte AS (...) {SELECT | INSERT | UPDATE | DELETE} ...`（CTE，支援多 CTE + 欄位清單；主體可為四種 DML，見 §1.1–1.3）
- [x] `CREATE TABLE`
- [x] `SET @var = value`（變數賦值）
- [~] `SET <option> {ON|OFF | value}`（session 選項：`SET NOCOUNT ON`/`XACT_ABORT OFF`、`SET IDENTITY_INSERT table ON`、取值型 `SET ROWCOUNT 100`/`SET DATEFORMAT mdy`/`SET LOCK_TIMEOUT 1000`）（`SqlSetOptionStatement { Option, Target, Value }`，`ParseSetOptionStatement` 排在變數賦值前、peek `=` 則 reset 落回 `SET @x = ...`；ON/OFF 用 `TryMatchOnOff`、其餘值走 `ParseArithmeticExpr().ToSql()`，第二個值存 `Target`（IDENTITY_INSERT）。未做多字 `SET TRANSACTION ISOLATION LEVEL`）
- [x] `WAITFOR {DELAY|TIME} 'time'`（`SqlWaitForStatement` + `SqlWaitForKind`；time 走 `ParseArithmeticExpr`。未做 `WAITFOR (RECEIVE ...)`）
- [ ] `GOTO label` / `label:`（標籤）
- [ ] `BULK INSERT` / `BACKUP` / `RESTORE` / `DBCC` / `KILL`（運維語句，低優先）
- [x] `EXEC sp_addextendedproperty ...`（僅此特定 SP）
- [x] `INSERT`（parser 可解析常用語法，細目見 §1.1：VALUES/多列/SELECT/EXEC/DEFAULT VALUES/TOP/hint/OUTPUT/DEFAULT 值/CTE 前綴。additive 擴充 `SqlInsertStatement`，builder 路徑不受影響）
- [x] `UPDATE`（parser 可解析：SET 多指派 / 複合指派 `+=` / `t.col` / `DEFAULT` 值 / `FROM`+JOIN / `WHERE` / `TOP` / table hint / `OUTPUT` / CTE 前綴，細目見 §1.2）
- [~] `DELETE`（parser 可解析：`[FROM] t` / 省略 FROM / 第二個 `FROM`+JOIN / `WHERE` / `TOP` / table hint / `OUTPUT` / CTE 前綴，細目見 §1.3。已大致完整）
- [x] `MERGE`（parser 可解析：TOP、target/source（含 hint）、`ON`、三種 WHEN、`AND` 過濾、UPDATE/DELETE/INSERT（含 DEFAULT VALUES）action、OUTPUT、OPTION、CTE 前綴、結尾 `;`，細目見 §1.4）
- [~] `ALTER TABLE`（ADD 欄位（多）/ ADD CONSTRAINT（含 `WITH CHECK/NOCHECK` 前綴）/ DROP COLUMN（多）/ DROP CONSTRAINT（多）/ ALTER COLUMN / `{ENABLE\|DISABLE} TRIGGER {ALL\|names}` / `{CHECK\|NOCHECK} CONSTRAINT {ALL\|names}`，細目見 §1.5。含 ADD 混合欄位+約束）；`ALTER VIEW/PROCEDURE/FUNCTION/TRIGGER` 已支援（見對應 CREATE 列）
- [~] `DROP ...`（支援 `DROP {TABLE|VIEW|PROCEDURE|FUNCTION|INDEX|TRIGGER|SCHEMA|DATABASE|SEQUENCE|TYPE} [IF EXISTS] name1[, ...]`，含 `DROP INDEX ix ON table`（`SqlDropStatement.OnTable`）；`SqlDropStatement` + `SqlDropObjectType` enum。未做舊式 `DROP INDEX table.idx`、多 `idx ON tbl` 對）
- [x] `TRUNCATE TABLE`（`SqlTruncateTableStatement`）
- [~] `CREATE VIEW` / `ALTER VIEW`（`{CREATE [OR ALTER]|ALTER} VIEW v [(cols)] [WITH opt[, ...]] AS <query> [WITH CHECK OPTION]`；body 走 `Parse_CteBodyStatement` 故支援 CTE/SELECT；AS 前 `WITH opt`（SCHEMABINDING/ENCRYPTION…，含多字 `EXECUTE AS x`/`RETURNS NULL ON NULL INPUT`/`CALLED ON NULL INPUT`）入 `Options`（共用 `Parse_WithOptionList`/`Parse_WithOption`）。`SqlCreateViewStatement`（`IsAlter` 旗標）。）
- [~] `CREATE INDEX`（支援 `CREATE [UNIQUE] [CLUSTERED|NONCLUSTERED] INDEX ix ON t (col [ASC|DESC], ...) [INCLUDE (cols)] [WHERE filter]`；重用 `ParseColumnsAscDesc`。`SqlCreateIndexStatement`。未做尾端 `WITH (options)`）
- [~] `CREATE PROCEDURE` / `ALTER PROCEDURE`（`{CREATE [OR ALTER]|ALTER} {PROCEDURE|PROC} name [([@p type [(size)] [= default] [OUTPUT]] , ...)] AS <body>`；AS 前 `WITH opt[, ...]`（ENCRYPTION/RECOMPILE…）入 `Options`；body 走 `Parse()`（單一語句／`BEGIN…END`）。含多字 `EXECUTE AS {CALLER|OWNER|SELF}`。`SqlCreateProcedureStatement`（`IsAlter` 旗標）+ `SqlProcedureParameter`。未做 `FOR REPLICATION`、無 BEGIN 的多裸語句 body）
- [~] `CREATE FUNCTION` / `ALTER FUNCTION`（scalar：`RETURNS type[(size)] AS <body>`；inline TVF：`RETURNS TABLE AS RETURN (select)`；multi-statement TVF：`RETURNS @t TABLE (col defs) AS BEGIN…RETURN END`（`ReturnTableVariable`/`ReturnTableColumns`，欄位重用 `ParseColumnDefinition`）。AS 前 `WITH opt[, ...]`（SCHEMABINDING…）入 `Options`。`SqlCreateFunctionStatement`（`IsAlter` 旗標），body 走 `Parse()`、return clause 抽 `ParseFunctionReturnClause`。含多字選項 `RETURNS NULL ON NULL INPUT`/`CALLED ON NULL INPUT`）
- [~] `CREATE TRIGGER` / `ALTER TRIGGER`（`SqlCreateTriggerStatement`（`IsAlter` 旗標）+ `SqlTriggerTiming`/`SqlTriggerEvent` enum；`{CREATE [OR ALTER]|ALTER} TRIGGER name ON target {FOR|AFTER|INSTEAD OF} {INSERT|UPDATE|DELETE}[, ...] AS <body>`；ON 後、timing 前 `WITH opt[, ...]`（ENCRYPTION…）入 `Options`；body 重用 `Parse()`。未做 `DDL/LOGON` trigger、`FOR EACH ROW`）
- [~] `CREATE SCHEMA`（`SqlCreateSchemaStatement`；`CREATE SCHEMA name [AUTHORIZATION owner]`。未做 inline 物件定義 / GRANT 子句）/ `CREATE DATABASE`（`SqlCreateDatabaseStatement`；`CREATE DATABASE name`。未做 `ON`/`LOG ON`/`COLLATE` 等選項）
- [~] `DECLARE`（`DECLARE @v type [(size)] [= value] [, ...]`；`@t TABLE (col defs)` 表變數（`IsTable`/`TableColumns`，欄位重用 `Parse_ParenthesizedColumnDefinitions`）；`{@c|name} CURSOR [FOR <select>]` 游標（`IsCursor`/`CursorSource`，`Parse_CursorDeclaration`）；`SqlDeclareStatement` + `SqlVariableDeclaration`。未做表變數內 table 約束、CURSOR 進階選項（SCROLL/STATIC…））
- [~] 游標操作 `OPEN` / `CLOSE` / `DEALLOCATE`（單一 `SqlCursorOperationStatement` + `SqlCursorOperation` enum）、`FETCH [NEXT|PRIOR|FIRST|LAST|ABSOLUTE n|RELATIVE n] [FROM] cur [INTO @v[, ...]]`（`SqlFetchStatement`）；`@@FETCH_STATUS` 等全域變數已可用於 WHILE 條件（見 §5）。未做 `GLOBAL` 游標
- [x] `IF / ELSE`（`SqlIfStatement`；條件用 `Parse_WhereExpression`、then/else 各為單一語句，body 可為 `BEGIN...END`）
- [x] `WHILE`（`SqlWhileStatement`；body 為單一語句／`BEGIN...END`）
- [x] `BREAK` / `CONTINUE`（單一 `SqlLoopControlStatement` + `SqlLoopControlAction` enum；關鍵字語句）
- [~] `BEGIN ... END`（`SqlBlockStatement`，以共用 `ParseStatementsUntil` 解析 body；`BEGIN TRY`/`BEGIN TRAN` 由各自 parser 在前處理）
- [~] `RETURN [expr]`（`SqlReturnStatement`；值走 `ParseArithmeticExpr`，bare RETURN 在 `END`/`;`/EOF 前不取值。未做 `BREAK`/`CONTINUE`）
- [x] `BEGIN TRY ... END TRY / BEGIN CATCH ... END CATCH`（`SqlTryCatchStatement`；try/catch body 共用 `ParseStatementsUntil("END","TRY")`/`("END","CATCH")`）
- [~] `BEGIN / COMMIT / ROLLBACK / SAVE TRANSACTION`（`SqlTransactionStatement` + `SqlTransactionAction` 單類別 enum；`BEGIN|SAVE TRAN[SACTION]`、`COMMIT|ROLLBACK [TRAN|TRANSACTION|WORK]`、選擇性交易名稱以 stop-set 擋後續語句關鍵字。未做 `BEGIN DISTRIBUTED`、`@var` 名稱、`WITH MARK`）
- [~] `EXEC`（一般預存程序）：`{EXEC|EXECUTE} proc [arg, ...]`；具名參數 `@p = val [OUTPUT]`（`SqlExecArgument`，positional 仍為裸運算式）；動態 SQL `EXEC ('sql' | @sql)`（`SqlExecStatement.DynamicSql`）。`Parse_ExecStatement` → `SqlExecStatement`。未做 `EXEC (...) AT linked_server`
- [x] `USE <db>`（`SqlUseStatement`；`USE database_name`）
- [x] `GO`（批次分隔）（`SqlGoStatement`；選擇性 `GO count`）
- [~] `GRANT / REVOKE / DENY`（單一 `SqlPermissionStatement` + `SqlPermissionAction` enum；`{GRANT|REVOKE|DENY} perm[, ...] [ON securable] {TO|FROM} principal[, ...] [WITH GRANT OPTION] [CASCADE]`。未做 `object_type::` 前綴、`GRANT OPTION FOR`、`AS grantor`、多字權限 `CREATE TABLE`）
- [x] `PRINT`（`SqlPrintStatement`；值走 `ParseArithmeticExpr`，支援字串／變數／`+` 串接）
- [~] `THROW`（`SqlThrowStatement`；bare `THROW`（CATCH 重拋）或 `THROW error_number, message, state`。未做 `;THROW` 強制分號語意）
- [~] `RAISERROR`（`SqlRaiseErrorStatement`；`RAISERROR (msg, severity, state [, args...]) [WITH opt[, ...]]`，多餘參數入 `Arguments`、WITH 選項入 `Options`。未做格式字串語意檢查）

### 1.1 INSERT 細目（完整 T-SQL 文法對照）

已支援（見 `ParseInsertSqlTest.cs`）：
- [x] `INSERT [INTO] t [(col_list)] VALUES (...)`（單列）
- [x] 多列 `VALUES (..), (..), (..)`
- [x] 省略欄位清單 / 省略 `INTO`
- [x] `INSERT INTO t [(cols)] SELECT ...`
- [x] `INSERT INTO t DEFAULT VALUES`
- [x] VALUES 內任意運算式（函式 / `NULL` / 算術 / CASE，走 `ParseArithmeticExpr`）
- [x] `INSERT TOP (n) [PERCENT] ...`（重用 `Parse_TopClause`，掛 `SqlInsertStatement.Top`）
- [x] `OUTPUT col [AS alias] [, ...] [INTO target [(cols)]]`（`SqlOutputClause` 掛 `Output`；欄位重用 `Parse_Column_Arithmetic` + AS-unwrap，刻意不解析 bare alias 以避開 VALUES 被當別名）
- [x] 目標 table hint `INSERT INTO t WITH (TABLOCK [, ...]) ...`（抽共用 `Parse_WithTableHints`，與 FROM table hint 同源；掛 `SqlInsertStatement.Withs`）
- [x] `VALUES` 列內 `DEFAULT` 關鍵字當值（如 `VALUES (1, DEFAULT)`；`SqlDefaultValue`，僅在 VALUES 列 `Parse_InsertRowValue` 解析，不影響全域 `ParseValue`）

未支援（依價值排序）：
- [x] `INSERT INTO t [(cols)] EXEC proc [args]`（`SqlExecStatement` 掛 `SqlInsertStatement.ExecSource`；EXEC 動態 SQL / 具名參數已於頂層 EXEC 補齊，見 §1 EXEC）
- [x] CTE 前綴 `WITH cte AS (...) INSERT ...`（`ParseWithCteStatement` 改用 `Parse_CteBodyStatement` dispatch SELECT/INSERT/UPDATE/DELETE，見 `ParseCteDmlTest.cs`）

### 1.2 UPDATE 細目（已實作，見 `ParseUpdateSqlTest.cs`）

> 沿用 INSERT 的成功模式：**additive 擴充、重用既有 helper、TDD 一項一 commit**。

**核心約束（與 INSERT 同）**：`SqlUpdateStatement` 目前是 builder 專用形狀
（`SetColumns : List<SqlSetColumn>`，`SqlSetColumn` 帶 `ColumnName`/`ParameterName`/`Value`，
`ToSql()` 固定輸出 `UPDATE t SET [col] = @p0`），被 `SqlUpdateExpressionBuilder` +
`SqlUpdateExpressionBuilderTest` 消費，**不可改形狀、不可動 `ToSql()`**。

**AST 設計（additive，parser 走新欄位、builder 走舊欄位）**：
- 新增 `SetClauses : List<SqlAssignExpr>`（`= []`）——**重用既有 `SqlAssignExpr { Left, Right }`**
  （`Parse_SelectColumns` 在 assign 情境已會產生它），不要動 builder 的 `SetColumns`，兩條路互不干擾。
- 新增 `Top : SqlTopClause?`、`Withs : List<ISqlExpression>`（`= []`）、
  `FromSources : List<ISqlExpression>`（`= []`）、`Where : ISqlExpression?`、`Output : SqlOutputClause?`。
- `SqlType.UpdateStatement` 已存在；`Visit_UpdateStatement` 目前只 `AddSqlExpression`，
  **要補走訪** `SetClauses` / `FromSources` / `Where` / `Output`（否則重演「子節點沒被走訪」雷）。

**Parser 整合**：`Parse()` dispatch 加 `ParseUpdateStatement`（INSERT 之後）。子句順序（T-SQL）：
`UPDATE [TOP (n) [PERCENT]] target [WITH (hints)] SET col=expr[, ...] [OUTPUT ...] [FROM src[, ...]] [WHERE ...]`
- TOP → 重用 `Parse_TopClause`
- target table → `Parse_SqlIdentifier`
- table hint → 重用 `Parse_WithTableHints`
- SET 清單 → `ParseWithComma` 解析 `col = expr`；左值用 `Parse_SqlIdentifier`（支援 `t.col`）、
  右值用 `ParseArithmeticExpr`，組成 `SqlAssignExpr`（或直接借 `Parse_Column_Arithmetic` 的 assign 路徑，先驗證再決定）
- OUTPUT → 重用 `Parse_OutputClause`（注意 UPDATE 的 OUTPUT 可引用 `inserted.`/`deleted.` 兩個偽資料表，欄位解析不變）
- FROM → 重用 `Parse_FromSources`（含 JOIN）
- WHERE → 重用 `Parse_WhereExpression`

實際實作：SET 左值用 `Parse_SqlIdentifier`、右值用共用 `Parse_ValueOrDefault`（由原
`Parse_InsertRowValue` 改名而來，INSERT 列值與 UPDATE SET 共用），組成 `SqlAssignExpr`。

**MVP 清單**：
- [x] `UPDATE t SET a = 1`（單一指派）
- [x] `UPDATE t SET a = 1, b = 'x'`（多指派）
- [x] `UPDATE t SET a = expr WHERE ...`
- [x] `UPDATE t SET t.a = s.b FROM t JOIN s ON ...`（UPDATE ... FROM）
- [x] `SET col = DEFAULT`（共用 `Parse_ValueOrDefault` → `SqlDefaultValue`）
**第二階段**：
- [x] `UPDATE TOP (n) [PERCENT] ...`
- [x] 目標 table hint `WITH (...)`
- [x] `OUTPUT col [INTO target]`（`inserted.`/`deleted.` 偽資料表）
- [x] 複合指派 `+= -= *= /= %= &= |= ^=`（`SqlAssignExpr.Operator`，default `=`；`Parse_AssignOperator`）
- [x] CTE 前綴 `WITH cte AS (...) UPDATE ...`（共用 `Parse_CteBodyStatement`）

### 1.3 DELETE 細目（已實作 MVP，見 `ParseDeleteSqlTest.cs`）

**AST 設計**：DELETE 無現成 AST，已**新增三處**（照 recipe）：
`SqlDeleteStatement` 類別 + `SqlType.DeleteStatement` enum 成員 + `Visit_DeleteStatement`（走訪 `FromSources`/`Where`/`Output`）。
欄位：`Top : SqlTopClause?`、`TableName : string`（`= string.Empty`）、`Withs`、
`FromSources : List<ISqlExpression>`（`= []`，第二個 FROM 的 join 來源）、`Where : ISqlExpression?`、`Output : SqlOutputClause?`。

**Parser 整合**：`Parse()` dispatch 加 `ParseDeleteStatement`。子句順序（T-SQL）：
`DELETE [TOP (n) [PERCENT]] [FROM] target [WITH (hints)] [OUTPUT ...] [FROM src[, ...]] [WHERE ...]`
- 注意 **兩個 FROM**：第一個 `FROM`（可省）後接 target；第二個 `FROM` 才是 join 來源。
  解析：`DELETE` → optional `TOP` → optional `FROM` → target 名（`Parse_SqlIdentifier`）→ hint → OUTPUT → optional 第二個 `FROM`（`Parse_FromSources`）→ WHERE。
- 其餘全部重用：`Parse_TopClause` / `Parse_WithTableHints` / `Parse_OutputClause` / `Parse_FromSources` / `Parse_WhereExpression`。

**MVP 清單**：
- [x] `DELETE FROM t`
- [x] `DELETE FROM t WHERE ...`
- [x] `DELETE t WHERE ...`（省略 FROM）
- [x] `DELETE t FROM t JOIN s ON ... WHERE ...`（DELETE ... 第二個 FROM）
**第二階段**：
- [x] `DELETE TOP (n) [PERCENT] ...`
- [x] 目標 table hint `WITH (...)`、`OUTPUT col [INTO]`（`deleted.`/`inserted.` 偽資料表）
- [x] CTE 前綴 `WITH cte AS (...) DELETE ...`（共用 `Parse_CteBodyStatement`）

**共同雷點（UPDATE/DELETE 動手前先想）**：
1. **ReservedWords**：`SET`、`FROM`、`WHERE`、`OUTPUT` 多為既有 / 位置順序消費，預期不需新增；
   但 `UPDATE`/`DELETE` 之後若有「值/別名位置會吃掉關鍵字」的情況再個別評估（參考 INSERT 的 OUTPUT 教訓）。
2. **builder 測試零回歸**：`SqlUpdateExpressionBuilderTest`（+ `ToSql()` 字串）必須保持綠燈——additive 設計就是為保它。
3. **OUTPUT 偽資料表**：UPDATE/DELETE 的 OUTPUT 可用 `deleted.` / `inserted.`，欄位解析沿用 `Parse_OutputClause`（不需改）。
4. **DELETE 雙 FROM** 是最易錯處，務必先寫「DELETE t FROM t JOIN s」測試守住。

### 1.4 MERGE 細目（已實作 MVP，見 `ParseMergeSqlTest.cs`）

> MERGE 是 DML 最大一塊。沿用既有手法：**重用 helper、TDD 一項一 commit**。

**AST 設計**（分離 action，避免單一類別塞各 action 的欄位 = code smell）：
- `SqlMergeStatement`：`Target`/`Source : ITableSource`、`OnCondition : ISqlExpression`、`WhenClauses : List<SqlMergeWhenClause>`。
- `SqlMergeWhenClause`：`MatchType`（enum `MergeMatchType { Matched, NotMatchedByTarget, NotMatchedBySource }`）、`AndCondition : ISqlExpression?`、`Action : ISqlMergeAction`。
- `ISqlMergeAction` 三實作：`SqlMergeUpdateAction { SetClauses : List<SqlAssignExpr> }`、`SqlMergeDeleteAction`、`SqlMergeInsertAction { Columns / Values / IsDefaultValues }`。
- 新增 5 個 `SqlType` 成員 + 5 個 `Visit_Merge*`（走訪子節點）。

**Parser 整合**：`Parse()` dispatch 加 `ParseMergeStatement`。文法（T-SQL）：
`MERGE [TOP(n)] [INTO] target [hints] [alias] USING source [alias] ON cond { WHEN ... THEN ... }+ [OUTPUT] [OPTION] ;`
- target/source → 重用 `Parse_TableSourceWithHints`（含 alias / 衍生表 / hint）
- ON / WHEN 的 `AND` 條件 → 重用 `Parse_WhereExpression`
- UPDATE action 的 SET → 重用 `Parse_UpdateSetClause`
- INSERT action 的 `(cols)` / `VALUES (row)` / `DEFAULT VALUES` → 重用 `Parse_ParenthesizedColumns` / `Parse_InsertValuesRow`
- WHEN 種類關鍵字用 `TryKeywords`（`MATCHED` / `NOT MATCHED [BY TARGET]` / `NOT MATCHED BY SOURCE`）

**MVP 清單**：
- [x] `MERGE [INTO] t [AS a] USING s [AS b] ON cond WHEN MATCHED THEN UPDATE SET ...`
- [x] `WHEN MATCHED THEN DELETE`
- [x] `WHEN NOT MATCHED [BY TARGET] THEN INSERT (cols) VALUES (...)`
- [x] `WHEN NOT MATCHED BY SOURCE THEN UPDATE/DELETE`
- [x] WHEN 的 `AND <condition>` 過濾（重用 `Parse_WhereExpression`，停在 `THEN`）
- [x] 結尾 `;`（可省）
- [x] 無 alias `MERGE Target USING Source ON ...`（靠 `USING` 入 `ReservedWords`）
**第二階段**：
- [x] `MERGE TOP (n) [PERCENT]`（重用 `Parse_TopClause`）
- [x] target table hint（重用 `Parse_TableSourceWithHints` → `Target.Withs`；註：hint 須在 alias 前、無 alias，hint+alias 並用受 `Parse_TableSourceWithHints` 順序限制）
- [x] `OUTPUT col [INTO]`（重用 `Parse_OutputClause`）、結尾 `OPTION (...)`（重用 `ParseOptionClause`）
- [x] `INSERT DEFAULT VALUES` action
- [x] CTE 前綴 `WITH cte AS (...) MERGE ...`（`Parse_CteBodyStatement`）
- [x] `OUTPUT $action`（pseudo-column；既有 field reader 整串擷取為 `SqlFieldExpr`，`Merge_with_output_action` 測試已驗證）

**實作雷點（已確認）**：
1. **`USING` 必須入 `ReservedWords`**：否則無 alias 的 `MERGE Target USING ...` 會把 `USING` 當 target 的 bare alias 吃掉（`ON`/`WHEN`/`THEN` 因條件 / 運算式不解析 alias 而安全，故未加）。
2. 多個 `WHEN` 子句用 `while (Try(Parse_MergeWhenClause...))` loop 收集。
3. action 全部重用：UPDATE→`Parse_UpdateSetClause`、INSERT→`Parse_ParenthesizedColumns`+`Parse_InsertValuesRow`、DELETE→無。

### 1.5 ALTER TABLE 細目（已實作 MVP，見 `ParseAlterTableSqlTest.cs`）

**AST**：`SqlAlterTableStatement { TableName, Action : ISqlAlterTableAction }`；action 分離成 5 類
（`SqlAlterTableAddColumns` / `SqlAlterTableAddConstraint` / `SqlAlterTableDropColumn` / `SqlAlterTableDropConstraint` / `SqlAlterTableAlterColumn`）。
**重用**：ADD 欄位 / ALTER COLUMN → `ParseColumnDefinition`；ADD CONSTRAINT → `ParseTableConstraint`（已含 `CONSTRAINT name` + PK/UNIQUE/FK/CHECK）。
**ADD 分流**：peek `CONSTRAINT`/`PRIMARY`/`UNIQUE`/`FOREIGN`/`CHECK` → 約束路徑，否則欄位路徑。

**MVP 清單**：
- [x] `ADD col type [, col2 type ...]`（單 / 多欄位，含 size / NULL）
- [x] `ADD CONSTRAINT name PRIMARY KEY (...)`（PK；FK/UNIQUE/CHECK 同走 `ParseTableConstraint`）
- [x] `DROP COLUMN col [, col2]`
- [x] `DROP CONSTRAINT name [, name2]`
- [x] `ALTER COLUMN col newtype [NULL|NOT NULL]`
**第二階段**：
- [x] `WITH CHECK / WITH NOCHECK ADD CONSTRAINT ...`（additive `SqlAlterTableAddConstraint.WithCheck : bool?`，前綴 `WITH CHECK`/`WITH NOCHECK` 在 ADD 前解析）
- [x] `CHECK / NOCHECK CONSTRAINT {ALL | name[, ...]}`（`SqlAlterTableCheckConstraint`）
- [x] `{ENABLE|DISABLE} TRIGGER {ALL | name[, ...]}`（`SqlAlterTableToggleTrigger`）
- [x] ADD 同時混合欄位 + 約束（`ADD c INT, CONSTRAINT ...`）（逗號 loop 收集欄位/約束；純欄位→`SqlAlterTableAddColumns`、單一純約束→`SqlAlterTableAddConstraint`、混合/多約束→`SqlAlterTableAddElements`，向後相容既有形狀）
- [ ] `ALTER COLUMN` 的 `ADD/DROP` 子選項（罕見）

---

## 2. SELECT 子句

- [x] `SELECT ALL` / `SELECT DISTINCT`
- [x] `TOP (n)` / `TOP n` / `PERCENT` / `WITH TIES`
- [x] 欄位清單、`*`、別名（`AS` 或空白）、`table.column`
- [x] 運算式欄位（算術 / CASE / 函式 / 純量子查詢）
- [x] `FROM` 資料表 + 別名
- [x] 衍生表（FROM 內子查詢）
- [x] 資料表值函式（table-valued function）作為來源
- [x] table hint `WITH (NOLOCK, INDEX(...))`
- [x] `CHANGETABLE (CHANGES ...)`
- [x] 逗號分隔多來源（舊式 cross join）
- [x] `WHERE`（AND / OR / NOT、比較、LIKE、IN、BETWEEN、IS [NOT] NULL、EXISTS）
- [x] `GROUP BY`（運算式清單）
- [x] `GROUP BY ROLLUP / CUBE / GROUPING SETS`、`GROUP BY ALL`
- [x] `HAVING`
- [x] `ORDER BY`（`ASC` / `DESC`）
- [x] `OFFSET n ROWS [FETCH NEXT m ROWS ONLY]`
- [x] `UNION` / `UNION ALL`
- [x] `INTERSECT` / `EXCEPT`
- [x] `PIVOT` / `UNPIVOT`
- [x] `FOR XML`（`PATH`、`AUTO`、`RAW [('elem')]`、`EXPLICIT`、`ROOT`）
- [x] `FOR JSON`（`AUTO` / `PATH`、`ROOT[('name')]`、`INCLUDE_NULL_VALUES`、`WITHOUT_ARRAY_WRAPPER`）
- [x] `SELECT ... INTO new_table`（含暫存表 `#temp`）
- [~] `OPTION (query hint)`（支援 bare hint、`MAXDOP n` 數值、括號參數 hint、多 hint；hint 名稱以通用方式收集，未逐一驗證合法 hint 清單）
- [x] `TABLESAMPLE [SYSTEM] (n [PERCENT|ROWS]) [REPEATABLE (seed)]`（掛在 `SqlTableSource.TableSample`，位於 alias 之後、`WITH (hints)` 之前）
- [x] UNION 後套用於整體結果的 top-level `ORDER BY`（掛在外層 `SelectStatement.OrderBy`；bare set operand 用 `asSetOperand` 旗標不吃尾端 ORDER BY，括號子查詢仍保留自身 ORDER BY）

---

## 3. JOIN 類型

- [x] `INNER JOIN`
- [x] `JOIN`（隱含 inner）
- [x] `LEFT JOIN` / `LEFT OUTER JOIN`
- [x] `RIGHT JOIN` / `RIGHT OUTER JOIN`
- [x] `FULL JOIN` / `FULL OUTER JOIN`
- [x] `CROSS JOIN`
- [x] `CROSS APPLY`
- [x] `OUTER APPLY`
- [x] `ON` 條件（含 AND/OR 複合條件）

---

## 4. 視窗函式 (OVER)

- [x] `OVER (PARTITION BY ...)`
- [x] `OVER (ORDER BY ...)`
- [x] `OVER (PARTITION BY ... ORDER BY ...)`
- [x] `RANK()` / `ROW_NUMBER()` / 等（一般函式 + `OVER`）
- [~] 視窗框架 `ROWS / RANGE BETWEEN ... PRECEDING/FOLLOWING/CURRENT ROW/UNBOUNDED`（含單一 bound 與 BETWEEN 兩種形式）
  - 註：frame 只掛在「泛用值 + `OVER`」路徑（聚合視窗函式 `SUM()`/`AVG()` 等）。`RANK()`/`ROW_NUMBER()` 走 `ParseRankClause` 獨立路徑、未加 frame——但排名函式在 T-SQL 本就不允許 frame，故為刻意不做、非遺漏。
- [N/A] 視窗框架 `EXCLUDE` 選項（`EXCLUDE CURRENT ROW / GROUP / TIES / NO OTHERS`）— SQL:2011 標準語法，**SQL Server 不支援**，不適用 T-SQL parser，不實作
- [x] `WITHIN GROUP (...)`（`STRING_AGG`、`PERCENTILE_CONT/DISC`；含多欄與 `ASC`/`DESC`）
- [~] 具名 `WINDOW` 子句（SQL Server 2022+；`SqlWindowClause`/`SqlWindowDefinition` 掛在 `SelectStatement.Window`，於 HAVING 後、ORDER BY 前）
  - 支援：`WINDOW name AS (PARTITION BY ... ORDER BY ... frame)`（多個定義）、`func() OVER name` 名稱參照（`SqlOverWindowName`）
  - 未支援（刻意延後）：`OVER (existing_window ORDER BY ...)` 行內延伸參照、定義間互相參照 `AS (existing_window ...)`、`RANK()`/`ROW_NUMBER()` 的 bare `OVER name`（走 `ParseRankClause` 獨立路徑，要求 `(`）

---

## 5. 運算式與述詞 (Expressions / Predicates)

- [x] 算術 `+ - * /`
- [x] 位元 `& | ^`、一元 `~`
- [x] 比較 `= <> != > < >= <=`
- [x] `LIKE` / `NOT LIKE`
- [x] `IN (value list)`
- [x] `IN (subquery)`（既有功能；2026-06-21 補測試驗證並加回歸守護）
- [x] `BETWEEN ... AND ...`
- [x] `IS NULL` / `IS NOT NULL`
- [x] `EXISTS (subquery)`
- [x] `CASE WHEN ... THEN ... ELSE ... END`
- [x] `CAST(x AS type)`
- [x] `CONVERT(...)` / 一般純量函式（以泛用函式呼叫解析）
- [x] 一元負號（negative value）
- [x] `NOT` 運算式
- [x] 括號運算式
- [x] `COLLATE`（運算式層級 `WHERE / ORDER BY ... COLLATE`，及欄位定義見 §6）
- [x] 全域/系統變數 `@@x`（如 `@@FETCH_STATUS`/`@@ROWCOUNT`/`@@ERROR`/`@@IDENTITY`；既有 field reader 整串擷取為 `SqlFieldExpr`，可用於 WHILE/WHERE/SELECT；2026-06-21 補回歸測試驗證）

---

## 6. CREATE TABLE 細節

- [x] 欄位定義（資料型別 + size）
- [x] `NULL` / `NOT NULL`
- [x] `IDENTITY`
- [x] `DEFAULT` 約束
- [x] 計算欄位（`AS expr [PERSISTED]`）
- [x] `PRIMARY KEY`（欄位層級與資料表層級、`CLUSTERED`/`NONCLUSTERED`、`WITH (FILLFACTOR = ...)`）
- [x] `UNIQUE`
- [x] `FOREIGN KEY ... REFERENCES ...`
- [x] `CONSTRAINT` 命名
- [x] `CHECK` 約束（欄位層級與資料表層級、含 `CONSTRAINT` 命名）
- [x] 欄位 `COLLATE`（如 `VARCHAR(50) COLLATE Latin1_General_CI_AS`）
- [x] 欄位/資料表註解（透過獨立的 `sp_addextendedproperty` 語句）

---

## 7. 資料型別 (Data types)

- [x] 數值：`BIGINT INT SMALLINT TINYINT BIT DECIMAL NUMERIC MONEY SMALLMONEY FLOAT REAL`
- [x] 日期時間：`DATE DATETIME DATETIME2 DATETIMEOFFSET TIME`
- [x] 字串：`CHAR VARCHAR TEXT NCHAR NVARCHAR NTEXT`（含 size / `MAX`）
- [x] 二進位：`BINARY VARBINARY IMAGE`
- [x] 其他：`UNIQUEIDENTIFIER XML CURSOR TIMESTAMP ROWVERSION HIERARCHYID GEOMETRY GEOGRAPHY SQL_VARIANT`

---

## 8. LINQ ↔ SQL（`LinqParser`，附帶能力）

- [x] `from ... in ...`
- [x] `join ... in ... on ... equals ...`（含 `into`）
- [x] 多重 `from`（含 `DefaultIfEmpty()`）
- [x] `where`（`&&` / `||`、比較運算子）
- [x] `orderby`（`ascending` / `descending`）
- [x] `select`（單一欄位 / `select new { ... }`）

---

## 維護建議優先序（未完成項目）

1. 🟢 控制流程（DECLARE（含 `@t TABLE`/`@c CURSOR`）/IF/WHILE/BEGIN…END/RETURN/TRY…CATCH/TRANSACTION/BREAK/CONTINUE/PRINT/THROW/RAISERROR/游標操作 OPEN·CLOSE·DEALLOCATE·FETCH 已完成）：`BEGIN DISTRIBUTED TRANSACTION`、`;THROW` 分號語意、CURSOR 進階選項（SCROLL/STATIC/GLOBAL）
2. 🟢 DDL（CREATE+ALTER VIEW/PROC/FUNCTION（含 multi-statement TVF）/TRIGGER（含單字+多字 `WITH` 選項）、SCHEMA/DATABASE、ALTER TABLE 第二階段（含 ADD 混合）已完成）：DATABASE/SCHEMA 進階選項（`ON`/`LOG ON`/`COLLATE`/inline 物件）
3. 🟢 DML 細項（EXEC 動態 SQL / 具名參數、MERGE `OUTPUT $action`、`DECLARE @t TABLE` 已完成）：`EXEC (...) AT linked_server`、表變數內 table 約束
4. 🟢 具名 `WINDOW` 子句的延伸：`OVER (existing_window ...)` 行內參照、定義間互相參照、RANK 路徑 bare `OVER name`（見 §4 註）

✅ 已完成：`SELECT ... INTO`（2026-06-20）、`GROUP BY ROLLUP/CUBE/GROUPING SETS`（2026-06-20）、`FOR JSON`（2026-06-21）、視窗框架 `ROWS/RANGE BETWEEN`（2026-06-21）、`WITHIN GROUP`（2026-06-21）、`GROUP BY ALL`（2026-06-21）、`OPTION (query hint)`（2026-06-21）、`CHECK` 約束（2026-06-21）、欄位 `COLLATE`（2026-06-21）、運算式 `COLLATE`（2026-06-21）、UNION 後 top-level `ORDER BY`（2026-06-21）、`TABLESAMPLE`（2026-06-21）、`FOR XML RAW/EXPLICIT`（2026-06-21）、具名 `WINDOW` 子句 MVP（2026-06-21）、`INSERT` 解析（MVP + TOP/OUTPUT/hint/DEFAULT 值，2026-06-21）、`UPDATE` 解析（SET/FROM/WHERE/TOP/hint/OUTPUT/DEFAULT，2026-06-21）、`DELETE` 解析（雙 FROM/WHERE/TOP/hint/OUTPUT，2026-06-21）、CTE 前綴接 INSERT/UPDATE/DELETE（2026-06-21）、`MERGE` 解析 MVP（INTO/USING/ON/三種 WHEN/AND/三種 action，2026-06-21）、`TRUNCATE TABLE` + `DROP`（多型別 + IF EXISTS + 多名稱，2026-06-21）、`ALTER TABLE`（ADD/DROP COLUMN、ADD/DROP CONSTRAINT、ALTER COLUMN，2026-06-21）、`CREATE VIEW`（OR ALTER / 欄位清單 / WITH CHECK OPTION，2026-06-21）、`CREATE INDEX`（UNIQUE/CLUSTERED/ASC-DESC/INCLUDE/filtered WHERE，2026-06-21）、`DROP INDEX ix ON table`（2026-06-21）、DML 收尾（MERGE CTE 前綴 + DEFAULT VALUES、UPDATE 複合指派 `+=`、`INSERT ... EXEC`，2026-06-21）、頂層 `EXEC proc [args]`（2026-06-21）、MERGE 第二階段（TOP/hint/OUTPUT/OPTION，2026-06-21）、控制流程（`DECLARE`/`IF…ELSE`/`WHILE`/`BEGIN…END`，2026-06-21）、`CREATE PROCEDURE`（OR ALTER / 參數含 default+OUTPUT / body 重用 Parse()，2026-06-21）、`RETURN` + `CREATE FUNCTION`（scalar + inline TVF，2026-06-21）、`BEGIN TRY…CATCH` + `TRANSACTION`（BEGIN/COMMIT/ROLLBACK/SAVE，2026-06-21）、`PRINT`/`THROW`/`RAISERROR` + `BREAK`/`CONTINUE`（2026-06-21）、`CREATE TRIGGER` + `USE`/`GO`（2026-06-21）、multi-statement TVF + `GRANT`/`REVOKE`/`DENY`（2026-06-21）、`CREATE SCHEMA`/`DATABASE` + ALTER TABLE 第二階段（WITH CHECK/NOCHECK ADD、CHECK/NOCHECK CONSTRAINT、ENABLE/DISABLE TRIGGER，2026-06-21）、DML 細項收尾（EXEC 動態 SQL / 具名參數、MERGE `OUTPUT $action`、`DECLARE @t TABLE`，2026-06-21）、`ALTER VIEW/PROCEDURE/FUNCTION/TRIGGER`（共用 `TryDefinitionLead`，`IsAlter` 旗標，2026-06-21）、DDL `WITH` 選項（VIEW/PROC/FUNCTION/TRIGGER，共用 `Parse_WithOptionList`）+ ALTER TABLE ADD 混合欄位+約束（`SqlAlterTableAddElements`，2026-06-21）、多字 WITH 選項（`EXECUTE AS`/`RETURNS NULL ON NULL INPUT`）+ `DECLARE {@c|name} CURSOR [FOR select]`（2026-06-21）、游標操作 `OPEN`/`CLOSE`/`DEALLOCATE`/`FETCH`（2026-06-21）、全域變數 `@@x` 回歸驗證（既有功能，2026-06-21）、`SET <option> {ON|OFF}` session 選項（`SqlSetOptionStatement`，含 `IDENTITY_INSERT table`，2026-06-21）、`WAITFOR DELAY/TIME` + SET 取值型（ROWCOUNT/DATEFORMAT…，2026-06-21）

> 更新規則：每完成一項，於對應 `[ ]` 改成 `[x]`（部分完成用 `[~]` 並註記），並更新「最後驗證」日期。
