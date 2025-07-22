T-SQL 語法樹（AST）模型功能需求規格書
1. 總覽
本文件定義了一個簡化但結構正確的 T-SQL 語法樹（AST）模型的功能需求。此模型專注於 LINQ-like 查詢語法，且查詢的開頭必須為 FROM 子句。其主要目的是提供一個標準化的結構，用於表示、解析或生成 T-SQL 查詢語句。

2. 核心概念
抽象語法樹 (AST)：將 T-SQL 查詢語句表示為樹狀結構，其中每個節點代表語句中的一個構造。

LINQ-like 語法：查詢結構應類似於語言整合查詢 (LINQ) 的表達方式，強調從來源開始。

模組化設計：每個 T-SQL 子句應被定義為獨立的介面或型別，以便於組合和擴展。

條件表達式樹：WHERE 和 ON 子句中的條件將被細分為更精確的樹狀結構，包含運算元和運算符。

3. 功能需求細項
3.1. 根介面：TsqlAst
目的：作為整個 T-SQL 查詢 AST 的頂層容器。

必備屬性：

from: 必須包含一個 FromClause，定義查詢的來源。

select: 必須包含一個 SelectClause，定義要選擇的欄位。

可選屬性：

joins: 可選，一個 JoinClause 陣列，定義資料表連接。

where: 可選，一個 WhereClause，定義篩選條件。

groupBy: 可選，一個 GroupByClause，定義分組欄位。

orderBy: 可選，一個 OrderByClause 陣列，定義排序欄位。

3.2. From 子句 (FromClause)
目的：定義查詢的來源資料表或視圖。

屬性：

tableName: 字串型別，表示來源資料表的名稱（必填）。

alias: 可選字串型別，表示資料表的別名。

3.3. Join 子句 (JoinClause)
目的：定義資料表之間的連接操作。

屬性：

type: 列舉型別，表示連接類型，包含 'inner'、'left'、'right'、'full'。

tableName: 字串型別，表示要連接的資料表名稱（必填）。

alias: 可選字串型別，表示連接資料表的別名。

onCondition: Expression 型別，表示連接條件的表達式樹（必填）。

3.4. Where 子句 (WhereClause)
目的：定義查詢的篩選條件。

屬性：

condition: Expression 型別，表示篩選條件的邏輯表達式樹（必填）。

3.5. Select 子句 (SelectClause)
目的：定義要從查詢結果中選擇的欄位。

屬性：

fields: Field 物件的陣列，表示要選擇的欄位列表（必填）。

topN: 可選數字型別，表示 SELECT TOP N 的 N 值，用於限制返回的行數。

3.6. Field 定義
目的：定義單個欄位的名稱及其可選的別名。

屬性：

name: 字串型別，表示欄位的名稱（必填）。

alias: 可選字串型別，表示欄位的別名。

3.7. Group By 子句 (GroupByClause)
目的：定義用於分組的欄位。

屬性：

fields: 字串型別的陣列，表示用於分組的欄位名稱列表（必填）。

3.8. Order By 子句 (OrderByClause)
目的：定義查詢結果的排序方式。

屬性：

field: 字串型別，表示排序欄位的名稱（必填）。

direction: 可選列舉型別，表示排序方向，包含 'asc'（升序）或 'desc'（降序），預設為 'asc'。

3.9. 條件表達式樹 (Expression)
目的：定義複雜條件的樹狀結構。

型別：BinaryExpression | UnaryExpression | Literal | ColumnReference

3.9.1. 二元表達式 (BinaryExpression)
目的：表示兩個運算元之間的操作（例如 A = B, X AND Y）。

屬性：

type: 字串型別，固定為 'binary'。

operator: 字串型別，表示二元運算符（例如 '+', '-', '*', '/', '=', '<', '>', '<=', '>=', '<>', 'AND', 'OR', 'LIKE', 'IN' 等）。

left: Expression 型別，表示左側運算元。

right: Expression 型別，表示右側運算元。

3.9.2. 一元表達式 (UnaryExpression)
目的：表示單個運算元的操作（例如 NOT condition）。

屬性：

type: 字串型別，固定為 'unary'。

operator: 字串型別，表示一元運算符（例如 'NOT', '-'）。

operand: Expression 型別，表示運算元。

3.9.3. 字面值 (Literal)
目的：表示常數值。

屬性：

type: 字串型別，固定為 'literal'。

value: 任意型別，表示字面值（例如數字、字串、布林值、日期）。

3.9.4. 欄位引用 (ColumnReference)
目的：表示對資料庫欄位的引用。

屬性：

type: 字串型別，固定為 'column'。

name: 字串型別，表示欄位名稱，可包含表別名（例如 "u.UserId", "OrderId"）。

4. 資料型別定義
JoinType: 'inner' | 'left' | 'right' | 'full'

OrderDirection: 'asc' | 'desc'

5. 範例結構
以下是一個符合上述需求定義的 T-SQL AST 範例結構（以 JSON 形式表示，僅為說明結構，非實際程式碼）：

{
  "from": {
    "tableName": "Users",
    "alias": "u"
  },
  "joins": [
    {
      "type": "left",
      "tableName": "Orders",
      "alias": "o",
      "onCondition": {
        "type": "binary",
        "operator": "=",
        "left": {
          "type": "column",
          "name": "u.UserId"
        },
        "right": {
          "type": "column",
          "name": "o.UserId"
        }
      }
    }
  ],
  "where": {
    "condition": {
      "type": "binary",
      "operator": "AND",
      "left": {
        "type": "binary",
        "operator": ">",
        "left": {
          "type": "column",
          "name": "u.Age"
        },
        "right": {
          "type": "literal",
          "value": 25
        }
      },
      "right": {
        "type": "binary",
        "operator": ">=",
        "left": {
          "type": "column",
          "name": "o.OrderDate"
        },
        "right": {
          "type": "literal",
          "value": "2023-01-01"
        }
      }
    }
  },
  "select": {
    "fields": [
      {
        "name": "u.UserName",
        "alias": "Name"
      },
      {
        "name": "o.OrderId"
      },
      {
        "name": "o.TotalAmount"
      }
    ],
    "topN": 10
  },
  "groupBy": {
    "fields": [
      "u.UserName"
    ]
  },
  "orderBy": [
    {
      "field": "o.TotalAmount",
      "direction": "desc"
    },
    {
      "field": "u.UserName",
      "direction": "asc"
    }
  ]
}
