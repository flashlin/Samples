# TsSql

A TypeScript library for parsing LINQ-style queries (FROM-first syntax) and converting them to standard T-SQL.

## Features

- Parse LINQ-style queries with FROM-first syntax
- Convert to standard T-SQL with proper ordering
- Support for JOIN (INNER, LEFT, RIGHT, FULL)
- Support for WHERE, GROUP BY, HAVING, ORDER BY
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

## Development

```bash
# Install dependencies
pnpm install

# Run tests
pnpm test

# Build
pnpm build
```

