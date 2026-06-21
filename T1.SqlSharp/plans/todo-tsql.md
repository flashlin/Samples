# T1.SqlSharp — T-SQL 語法支援清單

> 用途：追蹤 parser 目前支援哪些 T-SQL 語法，方便維護與規劃。
> 圖例：`[x]` 已支援、`[ ]` 未支援、`[~]` 部分支援、`[N/A]` 不適用 T-SQL（不實作）。
> 最後驗證：2026-06-21（依 `T1.SqlSharp/ParserLit/SqlParser.cs`、`LinqParser.cs` 與測試實際比對）。
> 入口：`SqlParser.Parse()` dispatch 8 種頂層語句（WITH CTE / CREATE TABLE / SELECT / INSERT / UPDATE / DELETE / EXEC sp_addextendedproperty / SET）。

---

## 1. 頂層語句 (Top-level statements)

- [x] `SELECT`
- [x] `WITH cte AS (...) {SELECT | INSERT | UPDATE | DELETE} ...`（CTE，支援多 CTE + 欄位清單；主體可為四種 DML，見 §1.1–1.3）
- [x] `CREATE TABLE`
- [x] `SET @var = value`（變數賦值）
- [x] `EXEC sp_addextendedproperty ...`（僅此特定 SP）
- [~] `INSERT`（parser 可解析大部分常用語法，細目見 §1.1。additive 擴充 `SqlInsertStatement`（`Top`/`Withs`/`ValuesRows`/`SourceSelect`/`IsDefaultValues`/`Output`），builder 路徑不受影響。僅剩 `INSERT ... EXEC`、CTE 前綴未做）
- [~] `UPDATE`（parser 可解析：SET 多指派 / `t.col` / `DEFAULT` 值 / `FROM`+JOIN / `WHERE` / `TOP` / table hint / `OUTPUT` / CTE 前綴，細目見 §1.2。僅剩複合指派 `+=`）
- [~] `DELETE`（parser 可解析：`[FROM] t` / 省略 FROM / 第二個 `FROM`+JOIN / `WHERE` / `TOP` / table hint / `OUTPUT` / CTE 前綴，細目見 §1.3。已大致完整）
- [ ] `MERGE`
- [ ] `ALTER TABLE` / `ALTER ...`
- [ ] `DROP ...`
- [ ] `TRUNCATE TABLE`
- [ ] `CREATE VIEW` / `PROCEDURE` / `FUNCTION` / `INDEX` / `TRIGGER` / `SCHEMA` / `DATABASE`
- [ ] `DECLARE`
- [ ] `IF / ELSE`
- [ ] `WHILE`
- [ ] `BEGIN ... END`
- [ ] `BEGIN TRY ... END TRY / BEGIN CATCH ... END CATCH`
- [ ] `BEGIN / COMMIT / ROLLBACK TRANSACTION`
- [ ] `EXEC`（一般預存程序，非 sp_addextendedproperty）
- [ ] `USE <db>`
- [ ] `GO`（批次分隔）
- [ ] `GRANT / REVOKE / DENY`
- [ ] `PRINT` / `RAISERROR` / `THROW`

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
- [ ] `INSERT INTO t EXEC proc` / `EXEC ('sql')`（rowset 來源）
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
- [ ] 複合指派 `+= -= *= /=`（需新運算子，價值低）
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

1. 🟢 DML 收尾（小單點）：`INSERT ... EXEC`、UPDATE 複合指派 `+=`（見 §1.1/§1.2）
2. 🟢 `MERGE`（DML 最後一塊）
3. 🟢 具名 `WINDOW` 子句的延伸：`OVER (existing_window ...)` 行內參照、定義間互相參照、RANK 路徑 bare `OVER name`（見 §4 註）

✅ 已完成：`SELECT ... INTO`（2026-06-20）、`GROUP BY ROLLUP/CUBE/GROUPING SETS`（2026-06-20）、`FOR JSON`（2026-06-21）、視窗框架 `ROWS/RANGE BETWEEN`（2026-06-21）、`WITHIN GROUP`（2026-06-21）、`GROUP BY ALL`（2026-06-21）、`OPTION (query hint)`（2026-06-21）、`CHECK` 約束（2026-06-21）、欄位 `COLLATE`（2026-06-21）、運算式 `COLLATE`（2026-06-21）、UNION 後 top-level `ORDER BY`（2026-06-21）、`TABLESAMPLE`（2026-06-21）、`FOR XML RAW/EXPLICIT`（2026-06-21）、具名 `WINDOW` 子句 MVP（2026-06-21）、`INSERT` 解析（MVP + TOP/OUTPUT/hint/DEFAULT 值，2026-06-21）、`UPDATE` 解析（SET/FROM/WHERE/TOP/hint/OUTPUT/DEFAULT，2026-06-21）、`DELETE` 解析（雙 FROM/WHERE/TOP/hint/OUTPUT，2026-06-21）、CTE 前綴接 INSERT/UPDATE/DELETE（2026-06-21）

> 更新規則：每完成一項，於對應 `[ ]` 改成 `[x]`（部分完成用 `[~]` 並註記），並更新「最後驗證」日期。
