# SqlBoyLib

A fluent SQL query builder library for .NET that generates parameterized SQL queries using `sp_executesql` format.

## Installation

Add reference to SqlBoyLib project in your application.

## Usage

### Basic Example

```csharp
using SqlBoyLib;

// Define your entity class
public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; }
    public int Age { get; set; }
}

// Build a query
var query = Db.From<Customer>("Customers")
    .Where(x => x.Id > 1 && x.Name == "John")
    .OrderBy(x => x.Age)
    .Build();

// Get the SQL statement
string sql = query.ToExecuteSql();
```

### Output

```sql
EXEC sys.sp_executesql
  @stmt = N'SELECT * FROM Customers WHERE (Id > @p1 AND Name = @p2) ORDER BY Age ASC',
  @params = N'@p1 INT, @p2 NVARCHAR(MAX)',
  @p1 = 1, @p2 = N'John'
```

## Features

### Supported Operations

- **From<TEntity>(string tableName)**: Specify the table to query
- **Where(Expression<Func<TEntity, bool>>)**: Add WHERE conditions
- **OrderBy(Expression<Func<TEntity, object>>)**: Add ascending order
- **OrderByDescending(Expression<Func<TEntity, object>>)**: Add descending order
- **Build()**: Generate the final SqlQuery object

### Supported Operators in Where Clause

- Equality: `==`, `!=`
- Comparison: `>`, `>=`, `<`, `<=`
- Logical: `&&` (AND), `||` (OR)
- Negation: `!` (NOT)

### Supported Data Types

- `int`, `long`
- `string`
- `bool`
- `DateTime`
- `decimal`
- `Guid`

## Advanced Examples

### Complex Where Conditions

```csharp
var query = Db.From<Customer>("Customers")
    .Where(x => (x.Age > 18 && x.Age < 65) || x.Name == "Admin")
    .Build();
```

### Multiple OrderBy

```csharp
var query = Db.From<Customer>("Customers")
    .Where(x => x.Age > 18)
    .OrderBy(x => x.Name)
    .OrderByDescending(x => x.Age)
    .Build();
```

## API Reference

### Db Class

Static entry point for building queries.

#### Methods

- `From<TEntity>(string tableName)`: Creates a new query builder for the specified table

### SqlQueryBuilder<TEntity> Class

Fluent builder for constructing SQL queries.

#### Methods

- `Where(Expression<Func<TEntity, bool>> predicate)`: Adds a WHERE condition
- `OrderBy(Expression<Func<TEntity, object>> keySelector)`: Adds ascending order
- `OrderByDescending(Expression<Func<TEntity, object>> keySelector)`: Adds descending order
- `Build()`: Builds and returns the SqlQuery object

### SqlQuery Class

Represents the generated SQL query with parameters.

#### Properties

- `Statement`: The SQL SELECT statement
- `ParameterDefinitions`: Parameter type definitions
- `Parameters`: Dictionary of parameter names and values

#### Methods

- `ToExecuteSql()`: Generates the complete `sp_executesql` statement

