# T1.SqlSharp — T-SQL 語法支援清單

> 用途：追蹤 parser 目前支援哪些 T-SQL 語法，方便維護與規劃。
> 圖例：`[x]` 已支援、`[ ]` 未支援、`[~]` 部分支援、`[N/A]` 不適用 T-SQL（不實作）。
> 最後驗證：2026-06-21（依 `T1.SqlSharp/ParserLit/SqlParser.cs`、`LinqParser.cs` 與測試實際比對）。
> 入口：`SqlParser.Parse()` dispatch 6 種頂層語句（WITH CTE / CREATE TABLE / SELECT / INSERT / EXEC sp_addextendedproperty / SET）。

---

## 1. 頂層語句 (Top-level statements)

- [x] `SELECT`
- [x] `WITH cte AS (...) SELECT ...`（CTE，支援多 CTE + 欄位清單）
- [x] `CREATE TABLE`
- [x] `SET @var = value`（變數賦值）
- [x] `EXEC sp_addextendedproperty ...`（僅此特定 SP）
- [~] `INSERT`（parser 可解析大部分常用語法，細目見 §1.1。additive 擴充 `SqlInsertStatement`（`Top`/`Withs`/`ValuesRows`/`SourceSelect`/`IsDefaultValues`/`Output`），builder 路徑不受影響。僅剩 `INSERT ... EXEC`、CTE 前綴未做）
- [ ] `UPDATE`（同上，有 builder 無 parser）
- [ ] `DELETE`
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
- [ ] CTE 前綴 `WITH cte AS (...) INSERT ...`（需擴充 `SqlWithCte.Statement` 接受 INSERT，目前只接 SELECT）

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

1. 🟢 `INSERT` / `UPDATE` / `DELETE` 的「解析」能力（目前只有「產生」）
2. 🟢 具名 `WINDOW` 子句的延伸：`OVER (existing_window ...)` 行內參照、定義間互相參照、RANK 路徑 bare `OVER name`（見 §4 註）

✅ 已完成：`SELECT ... INTO`（2026-06-20）、`GROUP BY ROLLUP/CUBE/GROUPING SETS`（2026-06-20）、`FOR JSON`（2026-06-21）、視窗框架 `ROWS/RANGE BETWEEN`（2026-06-21）、`WITHIN GROUP`（2026-06-21）、`GROUP BY ALL`（2026-06-21）、`OPTION (query hint)`（2026-06-21）、`CHECK` 約束（2026-06-21）、欄位 `COLLATE`（2026-06-21）、運算式 `COLLATE`（2026-06-21）、UNION 後 top-level `ORDER BY`（2026-06-21）、`TABLESAMPLE`（2026-06-21）、`FOR XML RAW/EXPLICIT`（2026-06-21）、具名 `WINDOW` 子句 MVP（2026-06-21）

> 更新規則：每完成一項，於對應 `[ ]` 改成 `[x]`（部分完成用 `[~]` 並註記），並更新「最後驗證」日期。
