# T1.SqlSharp — T-SQL 語法支援清單

> 用途：追蹤 parser 目前支援哪些 T-SQL 語法，方便維護與規劃。
> 圖例：`[x]` 已支援、`[ ]` 未支援、`[~]` 部分支援、`[N/A]` 不適用 T-SQL（不實作）。
> 最後驗證：2026-06-21（依 `T1.SqlSharp/ParserLit/SqlParser.cs`、`LinqParser.cs` 與測試實際比對；578 測試全綠）。
> 入口：`SqlParser.Parse()` dispatch ~97 種頂層語句（WITH CTE / CREATE TABLE|VIEW|INDEX|PROCEDURE|FUNCTION|TRIGGER|SCHEMA|DATABASE / SELECT / INSERT / UPDATE / DELETE / MERGE / TRUNCATE / DROP / ALTER TABLE / EXEC sp_addextendedproperty / EXEC proc / DECLARE / BEGIN TRY…CATCH / TRANSACTION（BEGIN|COMMIT|ROLLBACK|SAVE）/ BEGIN…END / IF / WHILE / RETURN / PRINT / THROW / RAISERROR / BREAK / CONTINUE / OPEN|CLOSE|DEALLOCATE / FETCH / WAITFOR / USE / GO / GRANT|REVOKE|DENY / SET（變數賦值 + session 選項 ON/OFF/取值））。

---

## 1. 頂層語句 (Top-level statements)

- [x] `SELECT`
- [x] `WITH cte AS (...) {SELECT | INSERT | UPDATE | DELETE} ...`（CTE，支援多 CTE + 欄位清單；主體可為四種 DML，見 §1.1–1.3）
- [x] `CREATE TABLE`
- [x] `SET @var = value`（變數賦值）
- [~] `SET <option> {ON|OFF | value}`（session 選項：`SET NOCOUNT ON`/`XACT_ABORT OFF`、`SET IDENTITY_INSERT table ON`、取值型 `SET ROWCOUNT 100`/`SET DATEFORMAT mdy`/`SET LOCK_TIMEOUT 1000`；`SET TRANSACTION ISOLATION LEVEL {READ UNCOMMITTED|READ COMMITTED|REPEATABLE READ|SNAPSHOT|SERIALIZABLE}`，`ReadIsolationLevel` 處理 1–2 字 level）（`SqlSetOptionStatement { Option, Target, Value }`，`ParseSetOptionStatement` 排在變數賦值前、peek `=` 則 reset 落回 `SET @x = ...`；ON/OFF 用 `TryMatchOnOff`、其餘值走 `ParseArithmeticExpr().ToSql()`）
- [x] `WAITFOR {DELAY|TIME} 'time'`（`SqlWaitForStatement` + `SqlWaitForKind`；time 走 `ParseArithmeticExpr`。未做 `WAITFOR (RECEIVE ...)`）
- [x] `GOTO label`（`SqlGotoStatement`）/ `label:`（`SqlLabelStatement`；dispatch 置末、`ident :` 且非 `::`）
- [x] 運維語句：`BULK INSERT`（`SqlBulkInsertStatement`；`BULK INSERT t FROM 'file' [WITH (opt = val[, ...])]`）/ `DBCC`（`SqlDbccStatement`；`DBCC cmd [(args)] [WITH opt[, ...]]`）/ `BACKUP`·`RESTORE`（單一 `SqlBackupRestoreStatement`，`{BACKUP|RESTORE} {DATABASE|LOG|CERTIFICATE|MASTER KEY} [name] {TO|FROM} device[, ...] [WITH opt[, ...]]`，MASTER KEY 無名稱）/ `KILL session [WITH STATUSONLY]`（`SqlKillStatement`）
- [x] `EXEC sp_addextendedproperty ...`（僅此特定 SP）
- [x] `INSERT`（parser 可解析常用語法，細目見 §1.1：VALUES/多列/SELECT/EXEC/DEFAULT VALUES/TOP/hint/OUTPUT/DEFAULT 值/CTE 前綴。additive 擴充 `SqlInsertStatement`，builder 路徑不受影響）
- [x] `UPDATE`（parser 可解析：SET 多指派 / 複合指派 `+=` / `t.col` / `DEFAULT` 值 / `FROM`+JOIN / `WHERE` / `TOP` / table hint / `OUTPUT` / CTE 前綴，細目見 §1.2）
- [~] `DELETE`（parser 可解析：`[FROM] t` / 省略 FROM / 第二個 `FROM`+JOIN / `WHERE` / `TOP` / table hint / `OUTPUT` / CTE 前綴，細目見 §1.3。已大致完整）
- [x] `MERGE`（parser 可解析：TOP、target/source（含 hint）、`ON`、三種 WHEN、`AND` 過濾、UPDATE/DELETE/INSERT（含 DEFAULT VALUES）action、OUTPUT、OPTION、CTE 前綴、結尾 `;`，細目見 §1.4）
- [~] `ALTER TABLE`（ADD 欄位（多）/ ADD CONSTRAINT（含 `WITH CHECK/NOCHECK` 前綴）/ DROP COLUMN（多）/ DROP CONSTRAINT（多）/ ALTER COLUMN / `{ENABLE\|DISABLE} TRIGGER {ALL\|names}` / `{CHECK\|NOCHECK} CONSTRAINT {ALL\|names}`，細目見 §1.5。含 ADD 混合欄位+約束）；`ALTER VIEW/PROCEDURE/FUNCTION/TRIGGER` 已支援（見對應 CREATE 列）
- [~] `DROP ...`（核心型別走 `SqlDropObjectType` enum：`TABLE|VIEW|PROCEDURE|FUNCTION|INDEX|TRIGGER|SCHEMA|DATABASE|SEQUENCE|TYPE|SYNONYM|LOGIN|USER|ROLE`；擴充型別走 `SqlDropStatement.TypeName` 字串（多字 via `MatchMultiWordDropType`）：`CERTIFICATE`/`ASSEMBLY`/`CREDENTIAL`/`AGGREGATE`/`RULE`/`DEFAULT`/`QUEUE`/`SERVICE`/`CONTRACT`/`ENDPOINT`/`STATISTICS`、`MESSAGE TYPE`/`MASTER KEY`/`SYMMETRIC KEY`/`PARTITION FUNCTION`/`PARTITION SCHEME`/`FULLTEXT INDEX (ON t)`/`FULLTEXT CATALOG`/`SECURITY POLICY`/`EXTERNAL TABLE`/`EXTERNAL DATA SOURCE`/`EXTERNAL FILE FORMAT`/`COLUMN MASTER KEY`/`COLUMN ENCRYPTION KEY`；`[IF EXISTS]` + 多名稱 + `DROP INDEX ix ON table`。未做舊式 `DROP INDEX table.idx`）
- [x] `TRUNCATE TABLE`（`SqlTruncateTableStatement`）
- [~] `CREATE VIEW` / `ALTER VIEW`（`{CREATE [OR ALTER]|ALTER} VIEW v [(cols)] [WITH opt[, ...]] AS <query> [WITH CHECK OPTION]`；body 走 `Parse_CteBodyStatement` 故支援 CTE/SELECT；AS 前 `WITH opt`（SCHEMABINDING/ENCRYPTION…，含多字 `EXECUTE AS x`/`RETURNS NULL ON NULL INPUT`/`CALLED ON NULL INPUT`）入 `Options`（共用 `Parse_WithOptionList`/`Parse_WithOption`）。`SqlCreateViewStatement`（`IsAlter` 旗標）。）
- [~] `CREATE INDEX`（支援 `CREATE [UNIQUE] [CLUSTERED|NONCLUSTERED] [COLUMNSTORE] INDEX ix ON t [(col [ASC|DESC], ...)] [INCLUDE (cols)] [WHERE filter] [WITH (options)]`；含 clustered/nonclustered COLUMNSTORE（`IsColumnstore`，欄位可省）、`SPATIAL`（`IsSpatial`）；重用 `ParseColumnsAscDesc`。`SqlCreateIndexStatement`）
- [~] `CREATE PROCEDURE` / `ALTER PROCEDURE`（`{CREATE [OR ALTER]|ALTER} {PROCEDURE|PROC} name [([@p type [(size)] [= default] [OUTPUT]] , ...)] AS <body>`；AS 前 `WITH opt[, ...]`（ENCRYPTION/RECOMPILE…）入 `Options`；body 走 `Parse()`（單一語句／`BEGIN…END`）。含多字 `EXECUTE AS {CALLER|OWNER|SELF}`。`SqlCreateProcedureStatement`（`IsAlter` 旗標）+ `SqlProcedureParameter`。未做 `FOR REPLICATION`、無 BEGIN 的多裸語句 body）
- [~] `CREATE FUNCTION` / `ALTER FUNCTION`（scalar：`RETURNS type[(size)] AS <body>`；inline TVF：`RETURNS TABLE AS RETURN (select)`；multi-statement TVF：`RETURNS @t TABLE (col defs) AS BEGIN…RETURN END`（`ReturnTableVariable`/`ReturnTableColumns`，欄位重用 `ParseColumnDefinition`）。AS 前 `WITH opt[, ...]`（SCHEMABINDING…）入 `Options`。`SqlCreateFunctionStatement`（`IsAlter` 旗標），body 走 `Parse()`、return clause 抽 `ParseFunctionReturnClause`。含多字選項 `RETURNS NULL ON NULL INPUT`/`CALLED ON NULL INPUT`）
- [~] `CREATE TRIGGER` / `ALTER TRIGGER`（`SqlCreateTriggerStatement`（`IsAlter` 旗標）+ `SqlTriggerTiming`/`SqlTriggerEvent` enum；`{CREATE [OR ALTER]|ALTER} TRIGGER name ON target {FOR|AFTER|INSTEAD OF} {INSERT|UPDATE|DELETE}[, ...] AS <body>`；ON 後、timing 前 `WITH opt[, ...]`（ENCRYPTION…）入 `Options`；body 重用 `Parse()`。DDL/LOGON trigger：`ON {DATABASE|ALL SERVER}` + DDL 事件名（`DdlEvents` 字串清單，如 `CREATE_TABLE`/`LOGON`）已支援。未做 `FOR EACH ROW`）
- [~] `CREATE SCHEMA`（`SqlCreateSchemaStatement`；`CREATE SCHEMA name [AUTHORIZATION owner]`。未做 inline 物件定義 / GRANT 子句）/ `CREATE DATABASE`（`SqlCreateDatabaseStatement`；`CREATE DATABASE name [ON [PRIMARY] (filespec)[, ...]] [LOG ON (filespec)[, ...]] [COLLATE x]`，filespec 走 `ReadFileSpecList` 收集 `(K = V[, ...])`，值走 `ReadFileSpecValue` 支援帶單位 `SIZE = 10MB`/`FILEGROWTH = 5MB`（number + 緊接 unit 合併）。未做 FILEGROUP、`CONTAINMENT`/`WITH` 資料庫選項）
- [~] `DECLARE`（`DECLARE @v type [(size)] [= value] [, ...]`；`@t TABLE (col defs [, table constraints])` 表變數（`IsTable`/`TableColumns`/`TableConstraints`，新 `Parse_ParenthesizedTableElements` 同時收欄位與 table 約束 PK/UNIQUE/CHECK/FK，`IsTableConstraintStart` 分流）；`{@c|name} CURSOR [opt...] [FOR <select>]` 游標（`IsCursor`/`CursorSource`/`CursorOptions`，`Parse_CursorDeclaration`，選項 LOCAL/GLOBAL/SCROLL/STATIC/KEYSET/DYNAMIC/FAST_FORWARD/READ_ONLY… 走 `TryMatchCursorOption`）；`SqlDeclareStatement` + `SqlVariableDeclaration`。未做表變數內 table 約束、ISO 式前置 `INSENSITIVE SCROLL CURSOR`）
- [~] 游標操作 `OPEN` / `CLOSE` / `DEALLOCATE`（單一 `SqlCursorOperationStatement` + `SqlCursorOperation` enum）、`FETCH [NEXT|PRIOR|FIRST|LAST|ABSOLUTE n|RELATIVE n] [FROM] cur [INTO @v[, ...]]`（`SqlFetchStatement`）；`@@FETCH_STATUS` 等全域變數已可用於 WHILE 條件（見 §5）。未做 `GLOBAL` 游標
- [x] `IF / ELSE`（`SqlIfStatement`；條件用 `Parse_WhereExpression`、then/else 各為單一語句，body 可為 `BEGIN...END`）
- [x] `WHILE`（`SqlWhileStatement`；body 為單一語句／`BEGIN...END`）
- [x] `BREAK` / `CONTINUE`（單一 `SqlLoopControlStatement` + `SqlLoopControlAction` enum；關鍵字語句）
- [~] `BEGIN ... END`（`SqlBlockStatement`，以共用 `ParseStatementsUntil` 解析 body；`BEGIN TRY`/`BEGIN TRAN` 由各自 parser 在前處理）
- [x] `RETURN [expr]`（`SqlReturnStatement`；值走 `ParseArithmeticExpr`，bare RETURN 在 `END`/`;`/EOF 前不取值）；`BREAK`/`CONTINUE`（`SqlLoopControlStatement` + `SqlLoopControlAction` enum，`ParseLoopControlStatement`）
- [x] `BEGIN TRY ... END TRY / BEGIN CATCH ... END CATCH`（`SqlTryCatchStatement`；try/catch body 共用 `ParseStatementsUntil("END","TRY")`/`("END","CATCH")`）
- [~] `BEGIN / COMMIT / ROLLBACK / SAVE TRANSACTION`（`SqlTransactionStatement` + `SqlTransactionAction` 單類別 enum；`BEGIN|SAVE TRAN[SACTION]`、`COMMIT|ROLLBACK [TRAN|TRANSACTION|WORK]`、`BEGIN [DISTRIBUTED] TRAN[SACTION]`（`IsDistributed` 旗標）、選擇性交易名稱（含 `@var`，stop-set 擋後續語句關鍵字）、`WITH MARK ['desc']`（`WithMark`/`MarkDescription`）。未做 `DELAYED_DURABILITY`）
- [~] `EXEC`（一般預存程序）：`{EXEC|EXECUTE} proc [arg, ...]`；具名參數 `@p = val [OUTPUT]`（`SqlExecArgument`，positional 仍為裸運算式）；動態 SQL `EXEC ('sql' | @sql)`（`SqlExecStatement.DynamicSql`）；回傳值擷取 `EXEC @ret = proc args`（`ReturnVariable`，偵測 `@var =` 前綴，否則 reset 落回變數 EXEC）；`EXEC (...) AT linked_server`（`AtLinkedServer`）。`Parse_ExecStatement` → `SqlExecStatement`
- [x] `USE <db>`（`SqlUseStatement`；`USE database_name`）
- [x] `GO`（批次分隔）（`SqlGoStatement`；選擇性 `GO count`）
- [~] `GRANT / REVOKE / DENY`（單一 `SqlPermissionStatement` + `SqlPermissionAction` enum；`{GRANT|REVOKE|DENY} perm[, ...] [ON securable] {TO|FROM} principal[, ...] [WITH GRANT OPTION] [CASCADE]`。含 `ON class::securable` 前綴（`SecurableClass`，如 `OBJECT::dbo.Orders`）、多字權限（`VIEW DEFINITION`/`CREATE TABLE`，`ReadPermissionNames` 逐字收集至 `,`/`ON`/`TO`/`FROM`）、`REVOKE GRANT OPTION FOR ...`（`GrantOptionFor`）、結尾 `AS grantor`（`AsGrantor`）、欄位層級 `GRANT SELECT (col1, col2) ON ...`（`Columns`，permission 後 peek `(`））
- [x] `PRINT`（`SqlPrintStatement`；值走 `ParseArithmeticExpr`，支援字串／變數／`+` 串接）
- [x] `THROW`（`SqlThrowStatement`；bare `THROW`（CATCH 重拋）或 `THROW error_number, message, state`；前導 `;THROW`（`Parse()` 開頭 skip 前導 `;`，為通用語句分隔）已支援）
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
**重用**：ADD 欄位 / ALTER COLUMN → `ParseColumnDefinition`；ADD CONSTRAINT → `ParseTableConstraint`（已含 `CONSTRAINT name` + PK/UNIQUE/FK/CHECK + 具名 `DEFAULT val FOR col`，`SqlConstraintDefaultValue.ForColumn`）。
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
- [x] `ALTER TABLE t REBUILD [WITH (options)]`（`SqlAlterTableRebuild`）
- [x] `ALTER TABLE t SET (option = value[, ...])`（`SqlAlterTableSet`，如 `SYSTEM_VERSIONING = ON`；`ReadParenthesizedAssignmentList`）
- [x] `ALTER TABLE t SWITCH [PARTITION n] TO target [PARTITION m]`（`SqlAlterTableSwitch`）
- [x] `ALTER TABLE t {ADD PERIOD FOR SYSTEM_TIME (c1, c2) | DROP PERIOD FOR SYSTEM_TIME}`（`SqlAlterTablePeriod`，分流於 ADD/DROP COLUMN 前）
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
- [x] `WITH XMLNAMESPACES (['uri' AS prefix | DEFAULT 'uri'][, ...]) <select>` 前綴（`SqlXmlNamespacesStatement`，dispatch 置於 WITH CTE 前，非 XMLNAMESPACES 則 reset；URI 用 `ParseSqlQuotedString` 避免吃 `AS`）
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
- [x] `LIKE` / `NOT LIKE`（含 `ESCAPE 'c'`，`SqlConditionExpression.Escape`）
- [x] 量化比較 `{> | < | = | ...} {ALL | ANY | SOME} (subquery)`（`SqlQuantifiedExpr`，在 `Parse_ConditionExpr` 比較運算子右側偵測 quantifier + `(子查詢)`）
- [x] `IS DISTINCT FROM` / `IS NOT DISTINCT FROM`（`ComparisonOperator.IsDistinctFrom`/`IsNotDistinctFrom`，多字運算子排在 `IS`/`IS NOT` 前比對）
- [x] 變數指派 `SELECT @v = expr` / `UPDATE t SET @v = expr`（既有 `SqlAssignExpr` 路徑，補回歸測試）
- [x] Unicode 字串字面值 `N'...'`（既有 field/value reader 保留 `N` 前綴，補回歸測試）
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
- [x] 資料表選項：`) ON filegroup|ps(col)`、`TEXTIMAGE_ON fg`、`FILESTREAM_ON fg`、`WITH (option = value[, ...])`（`SqlCreateTableExpression.OnFileGroup`/`TextImageOn`/`WithOptions`，`Parse_CreateTableOptions`）
- [x] 時間性資料表元素 `PERIOD FOR SYSTEM_TIME (c1, c2)`（`Period`，於欄位/約束 loop 後解析；搭配 `WITH (SYSTEM_VERSIONING = ON)`）。未做欄位 `GENERATED ALWAYS AS ROW START/END`
- [x] CTAS `CREATE TABLE name [WITH (opt[, ...])] AS SELECT ...`（`SqlCreateTableExpression.AsSelect`；表名後非 `(` 即走 CTAS 路徑，重用 `ParseSelectStatement`）

---

## 7. 資料型別 (Data types)

- [x] 數值：`BIGINT INT SMALLINT TINYINT BIT DECIMAL NUMERIC MONEY SMALLMONEY FLOAT REAL`
- [x] 日期時間：`DATE DATETIME DATETIME2 DATETIMEOFFSET TIME`
- [x] 字串：`CHAR VARCHAR TEXT NCHAR NVARCHAR NTEXT`（含 size / `MAX`）
- [x] 二進位：`BINARY VARBINARY IMAGE`
- [x] 其他：`UNIQUEIDENTIFIER XML CURSOR TIMESTAMP ROWVERSION HIERARCHYID GEOMETRY GEOGRAPHY SQL_VARIANT`

---

## 7.5 尚未涵蓋的語句 / 語法（候選，多為較少用）

### 其他 DDL（CREATE/ALTER 物件）
- [x] `CREATE SEQUENCE`（`SqlCreateSequenceStatement`；name + `AS type` + `START WITH` + `INCREMENT BY` + `[NO] MINVALUE`/`[NO] MAXVALUE`/`[NO] CYCLE`/`CACHE [n]`/`NO CACHE`，`ParseSequenceClauses` clause loop 逐子句儲存值（`MinValue`/`MaxValue`/`CacheSize`/`IsCycle`/`IsNoCycle`/`IsCache`/`IsNoCache`/`IsNoMinValue`/`IsNoMaxValue`）；`ALTER SEQUENCE name {RESTART [WITH n] | INCREMENT BY n | ...}`（`SqlAlterSequenceStatement`，clause 值仍僅消費））
- [x] `CREATE TYPE`（`SqlCreateTypeStatement`；`CREATE TYPE name {FROM base[(size)] | AS TABLE (col defs)}`，表型別欄位重用 `Parse_ParenthesizedColumnDefinitions`。`DROP TYPE` 已在 enum）
- [x] `CREATE SYNONYM name FOR target`（`SqlCreateSynonymStatement`）/ `DROP SYNONYM`（`SqlDropObjectType.Synonym`）
- [x] `ALTER INDEX {ix|ALL} ON t {REBUILD | REORGANIZE | DISABLE | SET (opt = val[, ...])}`（`SqlAlterIndexStatement`，`SET` 選項走 `ReadParenthesizedAssignmentList`；REBUILD/REORGANIZE 支援 `PARTITION = {n|ALL}`（`ReadOptionalIndexPartition` → `Partition`）與 `WITH (opt = val[, ...])`（`ReadOptionalIndexWithOptions`，ToSql 對非 SET 動作前綴 `WITH`））
- [x] `ALTER DATABASE name {SET setting [value] | ADD FILE (spec) | ADD LOG FILE (spec) | MODIFY FILE (spec)}`（`SqlAlterDatabaseStatement` + `FileAction`/`FileSpec`，filespec 重用 `ReadFileSpecList`；SET 值走 `ReadActionTokens` 收完整剩餘 token，支援 `COMPATIBILITY_LEVEL = 150`、`SINGLE_USER WITH ROLLBACK IMMEDIATE` 等多 token 值）
- [x] `ALTER SCHEMA name TRANSFER [OBJECT::]obj`（`SqlAlterSchemaStatement`）
- [x] `ALTER AUTHORIZATION ON [class::]securable TO principal`（`SqlAlterAuthorizationStatement`，含 `OBJECT::`/`SCHEMA::` 等 class 前綴。未做 `TO SCHEMA OWNER`）
- [x] `CREATE CERTIFICATE name [FROM FILE = 'x'] [ENCRYPTION BY PASSWORD = 'x'] [WITH opt = val[, ...]]`（`SqlCreateCertificateStatement`，clause loop）
- [x] `CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'x'`（`SqlCreateMasterKeyStatement`）
- [x] `SETUSER ['user']`（`SqlSetUserStatement`）
- [x] PolyBase/外部物件：`CREATE EXTERNAL {TABLE (cols) | DATA SOURCE | FILE FORMAT} ... WITH (opt = val[, ...])`（單一 `SqlCreateExternalStatement` + `Kind`）
- [x] `CREATE SECURITY POLICY name ADD {FILTER|BLOCK} PREDICATE fn(args) ON table[, ...] [WITH (STATE = ON)]`（`SqlCreateSecurityPolicyStatement`）
- [x] `CREATE [DATABASE SCOPED] CREDENTIAL name WITH IDENTITY = 'x'[, SECRET = 'y']`（`SqlCreateCredentialStatement`，`IsDatabaseScoped`；dispatch 置於 CREATE DATABASE 前）
- [x] `CREATE ASSEMBLY name [AUTHORIZATION o] [FROM 'x'] [WITH opt[, ...]]`（`SqlCreateAssemblyStatement`）
- [x] `CREATE {RULE|DEFAULT} name AS expr`（單一 `SqlCreateRuleOrDefaultStatement` + `IsRule`）
- [x] `CREATE AGGREGATE name (params) RETURNS type [EXTERNAL NAME asm.cls]`（`SqlCreateAggregateStatement`）
- [x] Service Broker `CREATE QUEUE name [WITH (opt = val[, ...])]`（`SqlCreateQueueStatement`）/ `CREATE SERVICE name ON QUEUE q [(contract)]`（`SqlCreateServiceStatement`）/ `CREATE CONTRACT name (msg SENT BY X)`（`SqlCreateContractStatement`）/ `CREATE MESSAGE TYPE name [VALIDATION = X]`（`SqlCreateMessageTypeStatement`）
- [x] Service Broker DML `SEND ON CONVERSATION @h [MESSAGE TYPE mt] (@body)`（`SqlSendStatement`）/ `RECEIVE [TOP (n)] * FROM q`（`SqlReceiveStatement`）
- [x] `CREATE ENDPOINT name [STATE = x] AS protocol (opts)`（`SqlCreateEndpointStatement`）
- [x] Resource Governor `CREATE {WORKLOAD GROUP|RESOURCE POOL} name [WITH (opts)]`（單一 `SqlCreateResourceGovernorObjectStatement` + `Kind`）
- [x] Always Encrypted `CREATE COLUMN {MASTER KEY|ENCRYPTION KEY} name WITH (opts)`（`SqlCreateColumnKeyStatement` + `IsMasterKey`）
- [x] `READTEXT` / `WRITETEXT` / `UPDATETEXT`（單一 `SqlTextPointerStatement`，引數收集至 `;`/EOF）
- [x] 通用 `ALTER {ASSEMBLY|CERTIFICATE|SERVICE|QUEUE|ENDPOINT|CONTRACT|ROUTE|CREDENTIAL|RESOURCE GOVERNOR|PARTITION FUNCTION|PARTITION SCHEME|MESSAGE TYPE|SYMMETRIC KEY|MASTER KEY|FULLTEXT CATALOG|SERVER ROLE|REMOTE SERVICE BINDING|...} [name] <action>`（單一 `SqlAlterObjectStatement` + `Kind`/`Name`/`Action`，`AlterObjectKinds` 多字比對，dispatch 置於各專屬 ALTER 之後；action 經 `ReadActionTokens` 收集）
- [x] `{ADD|DROP} SIGNATURE TO obj BY ...`（`SqlSignatureStatement`，dispatch 置於 DROP 前）
- [x] `CREATE [PRIMARY] XML INDEX ix ON t (col)`（`SqlCreateIndexStatement.IsXml`/`IsPrimaryXml`）
- [x] Service Broker 對話 DML：`BEGIN DIALOG [CONVERSATION]` / `END CONVERSATION` / `MOVE CONVERSATION` / `GET CONVERSATION GROUP`（單一 `SqlConversationStatement` + `Operation`/`Handle`/`Action`；dispatch 置於 BEGIN…END/TRY/TRAN 前）
- [x] 通用 `CREATE {EVENT SESSION|EVENT NOTIFICATION|REMOTE SERVICE BINDING|APPLICATION ROLE|SERVER ROLE|AVAILABILITY GROUP|ROUTE} name <action>`（單一 `SqlCreateObjectStatement` + `CreateObjectKinds`，dispatch 置於各專屬 CREATE 之後）
- [x] `CREATE FULLTEXT CATALOG name [AS DEFAULT] [AUTHORIZATION owner]`（`SqlCreateFulltextCatalogStatement`）/ `CREATE FULLTEXT STOPLIST name [FROM SYSTEM STOPLIST|src]`（`SqlCreateFulltextStoplistStatement`）
- [x] `CREATE SYMMETRIC KEY name [WITH opt[, ...]] [ENCRYPTION BY ...]`（`SqlCreateSymmetricKeyStatement`）
- [x] `{OPEN|CLOSE} SYMMETRIC KEY name [DECRYPTION BY ...]` / `{OPEN|CLOSE} ALL SYMMETRIC KEYS`（`SqlSymmetricKeyStatement`；dispatch 置於游標 OPEN/CLOSE 前，非 SYMMETRIC 則 reset 落回游標操作）
- [x] `ALTER FULLTEXT INDEX ON t <action>`（`SqlAlterFulltextIndexStatement`，action 收集剩餘字詞如 `START FULL POPULATION`）
- [x] `ALTER SERVER CONFIGURATION SET <setting>`（`SqlAlterServerConfigurationStatement`，setting 收集字詞與 `=`）
- [~] 安全性物件：`CREATE {LOGIN | USER | ROLE}`（單一 `SqlCreatePrincipalStatement` + `SqlPrincipalKind`；ROLE `AUTHORIZATION`、USER `FOR LOGIN`、LOGIN `WITH PASSWORD =`/`FROM WINDOWS`）；`ALTER ROLE name {ADD|DROP} MEMBER member`（`SqlAlterRoleStatement`）；`ALTER {LOGIN|USER} name {ENABLE|DISABLE | WITH opt = val[, ...]}`（`SqlAlterPrincipalStatement`，`ReadAssignmentOptionList` 讀非括號 `K = V` 清單）；`DROP {LOGIN|USER|ROLE}`（見 §1 DROP）。未做 `ALTER ROLE ... WITH NAME =` 改名
- [~] `CREATE STATISTICS name ON t (cols)` / `UPDATE STATISTICS t [name]`（單一 `SqlStatisticsStatement`，`IsCreate` 旗標）
- [x] `CREATE FULLTEXT INDEX ON t (cols) [KEY INDEX ix] [ON catalog]`（`SqlCreateFulltextIndexStatement`；columns 走 `ReadCommaSeparatedIdentifiers`。未做欄位 `LANGUAGE` 子選項）
- [x] `CREATE PARTITION FUNCTION name (type) AS RANGE [LEFT|RIGHT] FOR VALUES (v[, ...])`（`SqlCreatePartitionFunctionStatement`）/ `CREATE PARTITION SCHEME name AS PARTITION pf [ALL] TO (fg[, ...])`（`SqlCreatePartitionSchemeStatement`）
- [x] `CREATE XML SCHEMA COLLECTION name AS '<xsd>'`（`SqlCreateXmlSchemaCollectionStatement`）
- [x] `CREATE INDEX` 尾端 `WITH (options)`（`SqlCreateIndexStatement.Options`，共用 `Parse_ParenthesizedOptionList` 讀 `K = V`/`K`）

### 批次 / 控制 / 運維
- [x] `GOTO label` / `label:`（標籤）（見 §1）
- [x] `CHECKPOINT [n]` / `RECONFIGURE [WITH OVERRIDE]` / `REVERT` / `SHUTDOWN`（單一 `SqlKeywordStatement` + 選擇性 `Argument`；CHECKPOINT 取數值、RECONFIGURE 取 `WITH OVERRIDE`）
- [x] 獨立 `{ENABLE|DISABLE} TRIGGER {ALL|name[, ...]} ON {table|DATABASE|ALL SERVER}`（`SqlToggleTriggerStatement`；與 ALTER TABLE 內的 toggle trigger 不同路徑）
- [x] `BULK INSERT` / `DBCC` / `BACKUP` / `RESTORE` / `KILL`（見 §1；`BACKUP`/`RESTORE` 共用 `SqlBackupRestoreStatement`，device/option 走 `ReadAssignmentOptionList`）

### SELECT 進階來源 / 述詞
- [x] 表值建構式作為來源 `FROM (VALUES (1),(2)) AS t(c)`（`SqlValuesTableSource { Rows, Alias, ColumnAliases }`，在 `Parse_FromGroupWithTableSources` 加 VALUES 分支，重用 `Parse_InsertValuesRows` + `Parse_ParenthesizedColumns`）
- [x] `OPENJSON` / `OPENROWSET` / `OPENQUERY` / `OPENXML` 作為來源（既有 TVF 來源路徑）；`OPENROWSET(BULK 'file', SINGLE_CLOB)`（`Parse_FunctionArgument` 處理 `BULK` 前綴；`Parse_FromTableSources` 空清單防呆，修復原 `tableSourcesExpr[0]` 崩潰）；`OPENJSON(...) WITH (col type ['path'] [AS JSON][, ...])` 明確 schema（`SqlFuncTableSource.JsonSchemaColumns`，在 `Parse_FromTableSource` 偵測 OPENJSON + peek WITH 後以 `ReadOpenJsonSchema` 解析，重用 `Parse_DataSize`/`ParseSqlQuotedString`）
- [x] 全文檢索述詞 `CONTAINS(...)` / `FREETEXT(...)`（既有泛用函式呼叫路徑，補回歸測試；`CONTAINSTABLE`/`FREETEXTTABLE` 走 TVF 來源）
- [x] 時間性資料表 `FROM t FOR SYSTEM_TIME {AS OF expr | ALL | FROM x TO y | BETWEEN x AND y | CONTAINED IN (x, y)}`（`SqlTableSource.ForSystemTime` 字串，在 `Parse_TableSourceWithHints` 於 alias 前以 `Parse_ForSystemTime` 解析；bound 值用 `ParseArithmetic_Primary` 避免吃掉 `AND`）
- [x] `AT TIME ZONE 'zone'`（運算式層級 postfix，`SqlAtTimeZoneExpr`，掛在 `Parse_Value_As_DataType` COLLATE 前）

### 運算式特例
- [x] `NEXT VALUE FOR sequence`（序列值運算式，`SqlNextValueForExpr`，hook 在 `ParseArithmetic_Primary` 開頭，可用於 SELECT/DEFAULT/VALUES。未做 `OVER (...)` 尾綴）
- [x] `JSON_OBJECT('k': value[, ...])` 冒號鍵值對（`Parse_FunctionArgument` 在函式參數遇 `:` 時組成 `SqlAssignExpr`（Operator=`:`）；`JSON_ARRAY` 走一般函式引數）
- [x] ODBC 跳脫 `{ fn ... }` / `{ d '...' }` / `{ t '...' }` / `{ ts '...' }` / `{ guid '...' }`（`SqlOdbcEscapeExpr { Keyword, Body }`，hook 在 `ParseArithmetic_Primary` 開頭偵測 `{`，body 走 `ParseArithmeticExpr`）
- [x] `$PARTITION.fn(...)`（既有泛用函式呼叫路徑已支援，`FunctionName` 為 `$PARTITION.fn`，2026-06-21 補回歸測試驗證）。`$IDENTITY` / `$ROWGUID` 未個別處理

> 註：`IIF`/`CHOOSE`/`COALESCE`/`NULLIF`/`ISNULL`/`TRY_CAST`/`TRY_CONVERT`/`PARSE`/`sp_executesql` 等已透過「泛用函式呼叫 / 一般 EXEC」涵蓋，不另列。

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

✅ 已完成：`SELECT ... INTO`（2026-06-20）、`GROUP BY ROLLUP/CUBE/GROUPING SETS`（2026-06-20）、`FOR JSON`（2026-06-21）、視窗框架 `ROWS/RANGE BETWEEN`（2026-06-21）、`WITHIN GROUP`（2026-06-21）、`GROUP BY ALL`（2026-06-21）、`OPTION (query hint)`（2026-06-21）、`CHECK` 約束（2026-06-21）、欄位 `COLLATE`（2026-06-21）、運算式 `COLLATE`（2026-06-21）、UNION 後 top-level `ORDER BY`（2026-06-21）、`TABLESAMPLE`（2026-06-21）、`FOR XML RAW/EXPLICIT`（2026-06-21）、具名 `WINDOW` 子句 MVP（2026-06-21）、`INSERT` 解析（MVP + TOP/OUTPUT/hint/DEFAULT 值，2026-06-21）、`UPDATE` 解析（SET/FROM/WHERE/TOP/hint/OUTPUT/DEFAULT，2026-06-21）、`DELETE` 解析（雙 FROM/WHERE/TOP/hint/OUTPUT，2026-06-21）、CTE 前綴接 INSERT/UPDATE/DELETE（2026-06-21）、`MERGE` 解析 MVP（INTO/USING/ON/三種 WHEN/AND/三種 action，2026-06-21）、`TRUNCATE TABLE` + `DROP`（多型別 + IF EXISTS + 多名稱，2026-06-21）、`ALTER TABLE`（ADD/DROP COLUMN、ADD/DROP CONSTRAINT、ALTER COLUMN，2026-06-21）、`CREATE VIEW`（OR ALTER / 欄位清單 / WITH CHECK OPTION，2026-06-21）、`CREATE INDEX`（UNIQUE/CLUSTERED/ASC-DESC/INCLUDE/filtered WHERE，2026-06-21）、`DROP INDEX ix ON table`（2026-06-21）、DML 收尾（MERGE CTE 前綴 + DEFAULT VALUES、UPDATE 複合指派 `+=`、`INSERT ... EXEC`，2026-06-21）、頂層 `EXEC proc [args]`（2026-06-21）、MERGE 第二階段（TOP/hint/OUTPUT/OPTION，2026-06-21）、控制流程（`DECLARE`/`IF…ELSE`/`WHILE`/`BEGIN…END`，2026-06-21）、`CREATE PROCEDURE`（OR ALTER / 參數含 default+OUTPUT / body 重用 Parse()，2026-06-21）、`RETURN` + `CREATE FUNCTION`（scalar + inline TVF，2026-06-21）、`BEGIN TRY…CATCH` + `TRANSACTION`（BEGIN/COMMIT/ROLLBACK/SAVE，2026-06-21）、`PRINT`/`THROW`/`RAISERROR` + `BREAK`/`CONTINUE`（2026-06-21）、`CREATE TRIGGER` + `USE`/`GO`（2026-06-21）、multi-statement TVF + `GRANT`/`REVOKE`/`DENY`（2026-06-21）、`CREATE SCHEMA`/`DATABASE` + ALTER TABLE 第二階段（WITH CHECK/NOCHECK ADD、CHECK/NOCHECK CONSTRAINT、ENABLE/DISABLE TRIGGER，2026-06-21）、DML 細項收尾（EXEC 動態 SQL / 具名參數、MERGE `OUTPUT $action`、`DECLARE @t TABLE`，2026-06-21）、`ALTER VIEW/PROCEDURE/FUNCTION/TRIGGER`（共用 `TryDefinitionLead`，`IsAlter` 旗標，2026-06-21）、DDL `WITH` 選項（VIEW/PROC/FUNCTION/TRIGGER，共用 `Parse_WithOptionList`）+ ALTER TABLE ADD 混合欄位+約束（`SqlAlterTableAddElements`，2026-06-21）、多字 WITH 選項（`EXECUTE AS`/`RETURNS NULL ON NULL INPUT`）+ `DECLARE {@c|name} CURSOR [FOR select]`（2026-06-21）、游標操作 `OPEN`/`CLOSE`/`DEALLOCATE`/`FETCH`（2026-06-21）、全域變數 `@@x` 回歸驗證（既有功能，2026-06-21）、`SET <option> {ON|OFF}` session 選項（`SqlSetOptionStatement`，含 `IDENTITY_INSERT table`，2026-06-21）、`WAITFOR DELAY/TIME` + SET 取值型（ROWCOUNT/DATEFORMAT…，2026-06-21）、批次拉高完成度 10 項（`SET TRANSACTION ISOLATION LEVEL`、`BEGIN DISTRIBUTED TRAN`、`GOTO`/label、`CHECKPOINT`/`RECONFIGURE`/`REVERT`、`CREATE SEQUENCE`、`NEXT VALUE FOR`、`CREATE TYPE`、`CREATE/DROP SYNONYM`、`ALTER INDEX`、`FROM (VALUES…)`，2026-06-21）、拉高完成度 11–20 項（`OPENJSON`/`OPENQUERY` 來源驗證、`AT TIME ZONE`、`CONTAINS`/`FREETEXT` 驗證、`ALTER DATABASE SET`、`ALTER SCHEMA TRANSFER`、`CREATE LOGIN/USER/ROLE`、`CREATE/UPDATE STATISTICS`、`CREATE INDEX WITH(options)`、`DECLARE CURSOR` 選項、`GRANT ON class::`，2026-06-21）、完成度補強批次 1–5（`GRANT` 多字權限（`ReadPermissionNames`）、前導 `;THROW`（`Parse()` skip 前導 `;`）、`ALTER ROLE ADD/DROP MEMBER`、`ALTER LOGIN/USER`（ENABLE/DISABLE/WITH）、`DROP LOGIN/USER/ROLE`，2026-06-21）、完成度補強批次 6–10（`DBCC`、`BULK INSERT`、`ALTER SEQUENCE`、`CREATE DATABASE` ON/LOG ON/COLLATE、`FOR SYSTEM_TIME`，2026-06-21）、完成度補強批次 11–15（`KILL`、`BACKUP`/`RESTORE`、`OPENJSON WITH (schema)`、`REVOKE GRANT OPTION FOR`/`AS grantor`，2026-06-21）、完成度補強批次 16–20（ODBC 跳脫 `{ fn|d|t|ts }`、`$PARTITION` 回歸驗證、`CREATE FULLTEXT INDEX`、`CREATE PARTITION FUNCTION/SCHEME`、`CREATE XML SCHEMA COLLECTION`，2026-06-21）、拉高完成度第二輪 30 項（A 批：GRANT 欄位層級/`EXEC @ret=`/`DECLARE @t TABLE` 約束/具名 DEFAULT FOR/獨立 ENABLE-DISABLE TRIGGER/ALTER TABLE REBUILD·SET·SWITCH/COLUMNSTORE INDEX/ALTER AUTHORIZATION；B 批：ALTER INDEX SET/CREATE TABLE ON·TEXTIMAGE_ON·WITH/PERIOD FOR SYSTEM_TIME/EXEC AT linked server/JSON_OBJECT 冒號/CREATE CERTIFICATE·MASTER KEY/SETUSER/SPATIAL INDEX/BACKUP CERTIFICATE；C 批：CREATE FULLTEXT CATALOG·STOPLIST/CREATE·OPEN·CLOSE SYMMETRIC KEY/ALTER FULLTEXT INDEX/ALTER SERVER CONFIGURATION/CTAS/WITH XMLNAMESPACES/CHECKPOINT·RECONFIGURE 參數/OPENROWSET BULK 崩潰修復，2026-06-21）

> 更新規則：每完成一項，於對應 `[ ]` 改成 `[x]`（部分完成用 `[~]` 並註記），並更新「最後驗證」日期。
