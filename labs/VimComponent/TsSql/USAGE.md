# TsSql 使用說明

## 快速開始

### 安裝依賴

```bash
cd TsSql
pnpm install
```

### 開發模式

```bash
pnpm run dev
```

瀏覽器會自動開啟 `index.html` 展示 demo 頁面。

### 執行測試

```bash
# 執行所有測試
pnpm test

# 執行測試並顯示 UI
pnpm run test:ui
```

### 建置

```bash
pnpm run build
```

建置結果會輸出到 `dist/` 資料夾。

## API 使用範例

### 基本使用流程

```typescript
import { LinqParser, LinqToTSqlConverter, TSqlFormatter } from 'tssql';

// 1. 建立 parser
const parser = new LinqParser();

// 2. 解析 LINQ 查詢 (FROM-first 語法)
const parseResult = parser.parse(`
  FROM users u
  WHERE u.age > 18
  SELECT u.name, u.email
`);

// 3. 檢查是否有錯誤
if (parseResult.errors.length > 0) {
  console.log('Parse errors:', parseResult.errors);
}

// 4. 轉換為 T-SQL expression
const converter = new LinqToTSqlConverter();
const tsqlQuery = converter.convert(parseResult.result);

// 5. 格式化為 SQL 字串
const formatter = new TSqlFormatter();
const sql = formatter.format(tsqlQuery);

console.log(sql);
// 輸出:
// SELECT u.name, u.email
// FROM users u
// WHERE u.age > 18
```

### 支援的語法

#### 1. FROM 子句 (必須在開頭)

```typescript
FROM users
FROM users u  // 帶別名
FROM users AS u  // 使用 AS 關鍵字
```

#### 2. JOIN 子句 (支援多個)

```typescript
FROM users u
JOIN orders o ON u.id = o.user_id  // INNER JOIN

FROM users u
LEFT JOIN orders o ON u.id = o.user_id  // LEFT JOIN

FROM users u
RIGHT JOIN orders o ON u.id = o.user_id  // RIGHT JOIN

FROM users u
FULL JOIN orders o ON u.id = o.user_id  // FULL JOIN
```

#### 3. WHERE 子句 (支援多個,會用 AND 連接)

```typescript
FROM users
WHERE age > 18
WHERE status = 1
// 轉換後: WHERE age > 18 AND status = 1

FROM users
WHERE age > 18 AND status = 1 OR role = 2  // 複雜條件
```

#### 4. GROUP BY 子句 (支援多個欄位)

```typescript
FROM orders
GROUP BY customer_id
GROUP BY order_date
// 轉換後: GROUP BY customer_id, order_date
```

#### 5. HAVING 子句

```typescript
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 5
SELECT customer_id, COUNT(*)
```

#### 6. ORDER BY 子句 (支援多個欄位)

```typescript
FROM users
ORDER BY name ASC, age DESC
SELECT name, age
```

#### 7. SELECT 子句 (必須在最後)

```typescript
SELECT *  // 選擇所有欄位
SELECT name, email  // 選擇特定欄位
SELECT name AS username, email AS user_email  // 使用別名
SELECT DISTINCT country  // DISTINCT
SELECT COUNT(*), SUM(total), AVG(amount)  // 聚合函數
```

### 完整範例

```typescript
const complexQuery = `
  FROM users u
  LEFT JOIN orders o ON u.id = o.user_id
  LEFT JOIN products p ON o.product_id = p.id
  WHERE u.age > 18
  WHERE u.status = 1
  WHERE p.price > 100
  GROUP BY u.id, u.name
  HAVING COUNT(o.id) > 0
  ORDER BY u.name ASC
  ORDER BY COUNT(o.id) DESC
  SELECT u.name, COUNT(o.id) AS order_count, SUM(p.price) AS total_spent
`;

const parseResult = parser.parse(complexQuery);
const tsqlQuery = converter.convert(parseResult.result);
const sql = formatter.format(tsqlQuery);

console.log(sql);
/*
輸出:
SELECT u.name, COUNT(o.id) AS order_count, SUM(p.price) AS total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
LEFT JOIN products p ON o.product_id = p.id
WHERE u.age > 18
  AND u.status = 1
  AND p.price > 100
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY u.name ASC, COUNT(o.id) DESC
*/
```

### 錯誤處理

Parser 會保留已成功解析的部分,即使遇到錯誤也能產生部分結果:

```typescript
const result = parser.parse('FROM users WHERE SELECT name'); // 錯誤的語法

// result.result 包含已成功解析的部分
console.log(result.result.from); // FromExpression
console.log(result.result.where); // 可能是 undefined 或部分解析

// result.errors 包含錯誤訊息
result.errors.forEach(error => {
  console.log(error.toString());
  // 輸出: Parse error at line 1, column 18 (position 18): ...
});
```

### 表達式類型

Library 提供完整的 expression 類型,可以用程式方式建立查詢:

```typescript
import {
  QueryExpression,
  FromExpression,
  SelectExpression,
  ColumnExpression,
  BinaryExpression,
  BinaryOperator
} from 'tssql';

// 程式化建立查詢
const query = new QueryExpression(
  new SelectExpression([
    { expression: new ColumnExpression('name') },
    { expression: new ColumnExpression('email') }
  ]),
  new FromExpression('users', 'u'),
  [],
  new WhereExpression(
    new BinaryExpression(
      new ColumnExpression('age', 'u'),
      BinaryOperator.GreaterThan,
      new LiteralExpression(18, 'number')
    )
  )
);

const sql = formatter.format(query);
```

## 專案結構

```
TsSql/
├── src/
│   ├── types/              # 基礎型別定義
│   │   ├── ExpressionType.ts
│   │   ├── BaseExpression.ts
│   │   ├── ExpressionVisitor.ts
│   │   ├── ParseError.ts
│   │   └── ParseResult.ts
│   ├── expressions/        # T-SQL expression 類別
│   │   ├── ColumnExpression.ts
│   │   ├── LiteralExpression.ts
│   │   ├── BinaryExpression.ts
│   │   ├── UnaryExpression.ts
│   │   ├── FunctionExpression.ts
│   │   ├── FromExpression.ts
│   │   ├── JoinExpression.ts
│   │   ├── WhereExpression.ts
│   │   ├── GroupByExpression.ts
│   │   ├── HavingExpression.ts
│   │   ├── OrderByExpression.ts
│   │   ├── SelectExpression.ts
│   │   └── QueryExpression.ts
│   ├── linqExpressions/    # LINQ expression 類別
│   │   ├── LinqFromExpression.ts
│   │   ├── LinqJoinExpression.ts
│   │   ├── LinqWhereExpression.ts
│   │   ├── LinqGroupByExpression.ts
│   │   ├── LinqHavingExpression.ts
│   │   ├── LinqOrderByExpression.ts
│   │   ├── LinqSelectExpression.ts
│   │   └── LinqQueryExpression.ts
│   ├── parser/             # 解析器
│   │   ├── TokenType.ts
│   │   ├── Tokenizer.ts
│   │   └── LinqParser.ts
│   ├── converters/         # 轉換器
│   │   ├── LinqToTSqlConverter.ts
│   │   └── TSqlFormatter.ts
│   └── index.ts            # 公開 API
├── tests/                  # 測試檔案
│   ├── parser.test.ts
│   ├── converter.test.ts
│   ├── formatter.test.ts
│   └── integration.test.ts
├── package.json
├── tsconfig.json
├── vite.config.ts
├── vitest.config.ts
└── index.html              # Demo 頁面
```

## 特色功能

1. **FROM-first 語法**: 查詢從 FROM 開始,更接近 C# LINQ 的風格
2. **多個 WHERE/JOIN/GROUP BY**: 支援多個相同子句,自動合併
3. **錯誤恢復**: Parser 遇到錯誤會保留已解析的部分
4. **Visitor Pattern**: 使用標準設計模式,易於擴展
5. **型別安全**: 完整的 TypeScript 型別定義
6. **格式化輸出**: 自動格式化為可讀的 T-SQL,關鍵字大寫
7. **完整測試**: 包含 parser、converter、formatter 和整合測試

## 注意事項

1. 查詢必須以 `FROM` 開頭
2. `SELECT` 必須在最後
3. 子句順序: FROM → JOIN → WHERE → GROUP BY → HAVING → ORDER BY → SELECT
4. 多個 WHERE 會自動用 AND 連接
5. 多個 GROUP BY 或 ORDER BY 會自動合併

