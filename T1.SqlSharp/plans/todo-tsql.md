# T1.SqlSharp — T-SQL 語法支援清單

> 用途：追蹤 parser 目前支援哪些 T-SQL 語法，方便維護與規劃。
> 圖例：`[x]` 已支援、`[ ]` 未支援、`[~]` 部分支援。
> 最後驗證：2026-06-20（依 `T1.SqlSharp/ParserLit/SqlParser.cs`、`LinqParser.cs` 與測試實際比對）。
> 入口：`SqlParser.Parse()` 只 dispatch 5 種頂層語句（WITH CTE / CREATE TABLE / SELECT / EXEC sp_addextendedproperty / SET）。

---

## 1. 頂層語句 (Top-level statements)

- [x] `SELECT`
- [x] `WITH cte AS (...) SELECT ...`（CTE，支援多 CTE + 欄位清單）
- [x] `CREATE TABLE`
- [x] `SET @var = value`（變數賦值）
- [x] `EXEC sp_addextendedproperty ...`（僅此特定 SP）
- [ ] `INSERT`（註：`SqlInsertExpressionBuilder` 可「產生」，但 parser 不能「解析」）
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
- [ ] `GROUP BY ROLLUP / CUBE / GROUPING SETS` / `GROUP BY ALL`
- [x] `HAVING`
- [x] `ORDER BY`（`ASC` / `DESC`）
- [x] `OFFSET n ROWS [FETCH NEXT m ROWS ONLY]`
- [x] `UNION` / `UNION ALL`
- [x] `INTERSECT` / `EXCEPT`
- [x] `PIVOT` / `UNPIVOT`
- [~] `FOR XML`（支援 `PATH`、`AUTO`、`ROOT`；未支援 `RAW`、`EXPLICIT`）
- [ ] `FOR JSON`
- [x] `SELECT ... INTO new_table`（含暫存表 `#temp`）
- [ ] `OPTION (query hint)`（如 `OPTION(RECOMPILE / MAXDOP n)`）
- [ ] `TABLESAMPLE`
- [ ] UNION 後套用於整體結果的 top-level `ORDER BY`（目前會被內層 select 吃掉）

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
- [ ] 視窗框架 `ROWS / RANGE BETWEEN ... PRECEDING/FOLLOWING/CURRENT ROW/UNBOUNDED`
- [ ] `WITHIN GROUP (...)`（`STRING_AGG`、`PERCENTILE_CONT/DISC`）
- [ ] 具名 `WINDOW` 子句

---

## 5. 運算式與述詞 (Expressions / Predicates)

- [x] 算術 `+ - * /`
- [x] 位元 `& | ^`、一元 `~`
- [x] 比較 `= <> != > < >= <=`
- [x] `LIKE` / `NOT LIKE`
- [x] `IN (value list)`
- [ ] `IN (subquery)`（未驗證／未支援）
- [x] `BETWEEN ... AND ...`
- [x] `IS NULL` / `IS NOT NULL`
- [x] `EXISTS (subquery)`
- [x] `CASE WHEN ... THEN ... ELSE ... END`
- [x] `CAST(x AS type)`
- [x] `CONVERT(...)` / 一般純量函式（以泛用函式呼叫解析）
- [x] 一元負號（negative value）
- [x] `NOT` 運算式
- [x] 括號運算式
- [ ] `COLLATE`

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
- [ ] `CHECK` 約束
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

1. 🟡 `GROUP BY ROLLUP / CUBE / GROUPING SETS`（報表彙總）
2. 🟡 `FOR JSON`（API 場景）
3. 🟡 視窗框架 `ROWS / RANGE BETWEEN`
4. 🟢 `INSERT` / `UPDATE` / `DELETE` 的「解析」能力（目前只有「產生」）
5. 🟢 `CHECK` 約束、`COLLATE`、`OPTION` query hint、`WITHIN GROUP`

✅ 已完成：`SELECT ... INTO`（2026-06-20）

> 更新規則：每完成一項，於對應 `[ ]` 改成 `[x]`（部分完成用 `[~]` 並註記），並更新「最後驗證」日期。
