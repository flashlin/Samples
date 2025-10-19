# TsSql

A TypeScript library for parsing LINQ-style queries (FROM-first syntax) and converting them to standard T-SQL.

## Features

- Parse LINQ-style queries with FROM-first syntax
- Convert to standard T-SQL with proper ordering
- Support for JOIN (INNER, LEFT, RIGHT, FULL)
- Support for WHERE, GROUP BY, HAVING, ORDER BY
- Support for table hints (WITH NOLOCK, etc.)
- Error recovery during parsing
- Formatted T-SQL output with uppercase keywords

## Installation

```bash
pnpm install tssql
```

## Usage

```typescript
import { LinqParser, LinqToTSqlConverter, TSqlFormatter } from 'tssql';

// Parse LINQ-style query
const parser = new LinqParser();
const parseResult = parser.parse('FROM users WHERE age > 18 SELECT name, email');

// Convert to T-SQL expression
const converter = new LinqToTSqlConverter();
const tsqlExpr = converter.convert(parseResult.result);

// Format as T-SQL string
const formatter = new TSqlFormatter();
const sql = formatter.format(tsqlExpr);

console.log(sql);
// Output:
// SELECT name, email
// FROM users
// WHERE age > 18
```

### Using Table Hints

```typescript
// WITH(NOLOCK) for read uncommitted isolation
const linq = `
  FROM users WITH(NOLOCK) u
  JOIN orders WITH(NOLOCK) o ON u.id = o.user_id
  WHERE u.age > 18
  SELECT u.name, COUNT(o.id) AS order_count
`;

const parseResult = parser.parse(linq);
const tsqlExpr = converter.convert(parseResult.result);
const sql = formatter.format(tsqlExpr);

console.log(sql);
// Output:
// SELECT u.name, COUNT(o.id) AS order_count
// FROM users WITH(NOLOCK) u
// INNER JOIN orders WITH(NOLOCK) o ON u.id = o.user_id
// WHERE u.age > 18
```

See [HINTS_EXAMPLE.md](./HINTS_EXAMPLE.md) for more examples and supported hints.

## Development

```bash
# Install dependencies
pnpm install

# Run tests
pnpm test

# Build
pnpm build
```

