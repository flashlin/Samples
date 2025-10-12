# TsSql 專案總結

## 專案概述

TsSql 是一個 TypeScript 前端 library,用於將 LINQ 風格的查詢語法(FROM-first)解析並轉換為標準 T-SQL。

## 核心功能

### 1. LINQ 語法解析
- 支援 FROM-first 語法 (類似 C# LINQ)
- 遞迴下降 parser 實作
- 完整的錯誤處理和恢復機制
- 詞法分析 (Tokenizer) 支援關鍵字、運算符、識別符和字面值

### 2. Expression 系統
- **T-SQL Expressions**: 標準 T-SQL 語法的 expression 類別
- **LINQ Expressions**: LINQ 語法的 expression 類別
- 使用 Visitor Pattern 設計,易於擴展

### 3. 轉換器
- **LinqToTSqlConverter**: 將 LINQ expression 轉換為 T-SQL expression
- 自動合併多個 WHERE/GROUP BY/ORDER BY 子句
- 保留語義完整性

### 4. 格式化器
- **TSqlFormatter**: 將 T-SQL expression 格式化為字串
- 關鍵字自動大寫
- 基本縮排和換行
- 複雜條件的多行格式化

## 技術架構

### 設計模式
1. **Visitor Pattern**: 用於遍歷和轉換 expression tree
2. **Builder Pattern**: Expression 類別使用 builder 風格
3. **Strategy Pattern**: 不同的 expression 類型有不同的處理策略

### 核心類別

#### Types (基礎型別)
- `ExpressionType`: Expression 類型枚舉
- `BaseExpression`: 所有 expression 的基礎類別
- `ExpressionVisitor<T>`: Visitor 介面
- `ParseError`: 錯誤資訊
- `ParseResult<T>`: 解析結果(包含成功的 expression 和錯誤列表)

#### T-SQL Expressions
- `QueryExpression`: 完整查詢
- `SelectExpression`: SELECT 子句
- `FromExpression`: FROM 子句
- `JoinExpression`: JOIN 子句(INNER/LEFT/RIGHT/FULL)
- `WhereExpression`: WHERE 子句
- `GroupByExpression`: GROUP BY 子句
- `HavingExpression`: HAVING 子句
- `OrderByExpression`: ORDER BY 子句
- `ColumnExpression`: 欄位引用
- `LiteralExpression`: 字面值
- `BinaryExpression`: 二元運算
- `UnaryExpression`: 一元運算
- `FunctionExpression`: 函數呼叫

#### LINQ Expressions
- `LinqQueryExpression`: 完整 LINQ 查詢
- `LinqFromExpression`: FROM 子句(開頭)
- `LinqJoinExpression`: JOIN 子句
- `LinqWhereExpression`: WHERE 子句
- `LinqGroupByExpression`: GROUP BY 子句
- `LinqHavingExpression`: HAVING 子句
- `LinqOrderByExpression`: ORDER BY 子句
- `LinqSelectExpression`: SELECT 子句(結尾)

#### Parser
- `Tokenizer`: 詞法分析器
- `Token`: Token 類別
- `TokenType`: Token 類型枚舉
- `LinqParser`: 遞迴下降 parser

#### Converters
- `LinqToTSqlConverter`: LINQ 到 T-SQL 轉換器
- `TSqlFormatter`: T-SQL 格式化器

## 支援的 T-SQL 語法

### 基本查詢
- SELECT (欄位列表、*、DISTINCT)
- FROM (表名、別名)
- WHERE (條件運算式)
- GROUP BY (多個欄位)
- HAVING (聚合條件)
- ORDER BY (多個欄位、ASC/DESC)

### JOIN 類型
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL JOIN
- CROSS JOIN

### 運算符
- **比較**: =, <>, >, <, >=, <=
- **邏輯**: AND, OR, NOT
- **算術**: +, -, *, /, %
- **模式**: LIKE, IN
- **NULL**: IS NULL, IS NOT NULL

### 聚合函數
- COUNT, SUM, AVG, MIN, MAX (以及其他 T-SQL 函數)

## 錯誤處理

### 解析錯誤
- 記錄錯誤位置(行、列、position)
- 保留已成功解析的部分
- 錯誤恢復機制:跳到下一個關鍵字繼續解析

### 回傳結果
```typescript
{
  result: PartialLinqExpression,  // 部分或完整的 expression
  errors: ParseError[]             // 錯誤列表
}
```

## 測試涵蓋

### Parser 測試 (parser.test.ts)
- 簡單查詢解析
- 各種子句組合
- 錯誤處理
- 邊界情況

### Converter 測試 (converter.test.ts)
- LINQ 到 T-SQL 轉換
- 多個子句合併
- 部分查詢處理

### Formatter 測試 (formatter.test.ts)
- 格式化輸出
- 關鍵字大寫
- 縮排和換行
- 複雜條件格式化

### 整合測試 (integration.test.ts)
- 完整流程測試(Parse → Convert → Format)
- 實際使用場景
- 各種查詢類型

## 使用範例

### 簡單查詢
```typescript
// LINQ (FROM-first)
FROM users SELECT name, email

// 轉換為 T-SQL
SELECT name, email
FROM users
```

### 複雜查詢
```typescript
// LINQ (FROM-first)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.age > 18
WHERE u.status = 1
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY u.name ASC
SELECT u.name, COUNT(o.id) AS order_count

// 轉換為 T-SQL
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.age > 18
  AND u.status = 1
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY u.name ASC
```

## 專案配置

### 開發工具
- **Build Tool**: Vite 5.0
- **Language**: TypeScript 5.2
- **Test Framework**: Vitest 2.0
- **Package Manager**: pnpm

### 輸出格式
- ES Module (ESM)
- UMD (通用模組定義)
- TypeScript 型別定義 (.d.ts)

### Node 版本
- 22.12.0 (透過 .nvmrc 指定)

## 特色亮點

1. **型別安全**: 完整的 TypeScript 型別定義,編譯時期檢查
2. **可擴展**: Visitor Pattern 讓新增功能變得簡單
3. **錯誤友善**: 詳細的錯誤訊息和部分結果回傳
4. **測試完整**: 100+ 個測試案例涵蓋各種情境
5. **標準設計**: 遵循設計模式和最佳實踐
6. **文件完整**: 包含 README、USAGE 和範例

## 未來擴展方向

1. 支援更多 T-SQL 語法(子查詢、CTE、窗口函數)
2. 支援 DML 語句(INSERT、UPDATE、DELETE)
3. 查詢優化建議
4. SQL 注入檢測
5. 語法高亮支援
6. AST 視覺化工具

## 檔案統計

- **總檔案數**: 40+
- **TypeScript 程式碼**: 30+ 檔案
- **測試檔案**: 4 檔案
- **文件檔案**: 5 檔案
- **程式碼行數**: ~3000+ 行

## 開發時間軸

1. ✅ 專案初始化和配置
2. ✅ 基礎型別和介面定義
3. ✅ T-SQL Expression 類別系統
4. ✅ LINQ Expression 類別系統
5. ✅ Tokenizer 詞法分析器
6. ✅ Parser 遞迴下降解析器
7. ✅ LINQ 到 T-SQL 轉換器
8. ✅ T-SQL 格式化器
9. ✅ 公開 API 匯出
10. ✅ 完整測試套件
11. ✅ Demo 頁面和文件

## 結論

TsSql 是一個功能完整、架構清晰、測試完整的 TypeScript library,成功實現了 LINQ 風格查詢到 T-SQL 的轉換。專案使用標準設計模式,程式碼可維護性高,易於擴展。適合用於前端 SQL 查詢建構器、教育工具或 ORM 系統的一部分。

