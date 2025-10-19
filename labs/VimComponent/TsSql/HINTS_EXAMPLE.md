# WITH Hints Support

TsSql 現在支援 SQL Server 的 table hints（例如 NOLOCK）。

## 語法

```typescript
// FROM clause with hints
FROM tableName WITH(hint1, hint2, ...) [alias]

// JOIN clause with hints  
JOIN tableName WITH(hint1, hint2, ...) [alias] ON condition
```

## 範例

### 基本用法

```typescript
import { LinqParser, LinqToTSqlConverter, TSqlFormatter } from '@mrbrain/t1-tssql';

const parser = new LinqParser();
const converter = new LinqToTSqlConverter();
const formatter = new TSqlFormatter();

// 解析 LINQ 查詢
const linq = 'FROM users WITH(NOLOCK) SELECT name, email';
const parseResult = parser.parse(linq);

// 轉換為 T-SQL
const tsqlQuery = converter.convert(parseResult.result);
const sql = formatter.format(tsqlQuery);

console.log(sql);
// 輸出: SELECT name, email
//       FROM users WITH(NOLOCK)
```

### 多個 Hints

```typescript
const linq = 'FROM users WITH(NOLOCK, READUNCOMMITTED) u SELECT u.name';
const parseResult = parser.parse(linq);
const tsqlQuery = converter.convert(parseResult.result);
const sql = formatter.format(tsqlQuery);

console.log(sql);
// 輸出: SELECT u.name
//       FROM users WITH(NOLOCK, READUNCOMMITTED) u
```

### JOIN with Hints

```typescript
const linq = `
  FROM users WITH(NOLOCK) u
  JOIN orders WITH(NOLOCK) o ON u.id = o.user_id
  SELECT u.name, o.total
`;

const parseResult = parser.parse(linq);
const tsqlQuery = converter.convert(parseResult.result);
const sql = formatter.format(tsqlQuery);

console.log(sql);
// 輸出: SELECT u.name, o.total
//       FROM users WITH(NOLOCK) u
//       INNER JOIN orders WITH(NOLOCK) o ON u.id = o.user_id
```

### 複雜查詢

```typescript
const linq = `
  FROM users WITH(NOLOCK) u
  LEFT JOIN orders WITH(NOLOCK, READUNCOMMITTED) o ON u.id = o.user_id
  WHERE u.age > 18
  GROUP BY u.id, u.name
  HAVING COUNT(o.id) > 0
  ORDER BY u.name ASC
  SELECT u.name, COUNT(o.id) AS order_count
`;

const parseResult = parser.parse(linq);
const tsqlQuery = converter.convert(parseResult.result);
const sql = formatter.format(tsqlQuery);

console.log(sql);
// 輸出: SELECT u.name, COUNT(o.id) AS order_count
//       FROM users WITH(NOLOCK) u
//       LEFT JOIN orders WITH(NOLOCK, READUNCOMMITTED) o ON u.id = o.user_id
//       WHERE u.age > 18
//       GROUP BY u.id, u.name
//       HAVING COUNT(o.id) > 0
//       ORDER BY u.name ASC
```

## 支援的 Hints

理論上支援所有 SQL Server 的 table hints，常見的包括：

- `NOLOCK` - 允許髒讀（dirty reads）
- `READUNCOMMITTED` - 等同於 NOLOCK
- `READCOMMITTED` - 預設的隔離級別
- `READPAST` - 跳過被鎖定的資料列
- `SERIALIZABLE` - 最高的隔離級別
- `SNAPSHOT` - 使用 snapshot 隔離
- `UPDLOCK` - 使用更新鎖定
- `XLOCK` - 使用排他鎖定
- `ROWLOCK` - 使用資料列層級鎖定
- `PAGLOCK` - 使用頁面層級鎖定
- `TABLOCK` - 使用資料表層級鎖定
- `HOLDLOCK` - 保持鎖定直到交易完成

## 注意事項

1. Hints 名稱會自動轉換為大寫
2. Hints 必須放在括號內，並用逗號分隔
3. Hints 可以與 alias 同時使用
4. 語法順序：`table WITH(hints) alias`

