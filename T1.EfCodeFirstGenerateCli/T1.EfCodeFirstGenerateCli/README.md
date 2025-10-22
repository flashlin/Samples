# T1.EfCodeFirstGenerateCli

Automatically generate Entity Framework Core Code First classes from your database schema. This MSBuild Task integrates seamlessly with your build process.

## Installation

```bash
dotnet add package T1.EfCodeFirstGenerateCli
```

After installation, the code generation task will run automatically during build:

```bash
dotnet build
```

## Quick Start

### Step 1: Create Connection String Configuration File

Create a `.db` file in your project root directory (e.g., `Test.db`):

```
# Comment lines start with # or //

# SQL Server example
Server=localhost;Database=MyDatabase;User Id=sa;Password=YourPassword;TrustServerCertificate=true

# MySQL example  
Server=localhost;Database=TestDb;Uid=root;Pwd=secret;
```

**Supported connection string formats:**
- SQL Server: Standard ADO.NET format
- MySQL: MySQL Connector/NET format

### Step 2: Build Your Project

The MSBuild Task will automatically scan `.db` files and generate code during build:

```bash
dotnet build
```

### Step 3: Use Generated Code

Generated code will be placed in the `Generated/` directory:

```
Generated/
├── Test.schema          # Schema cache file
└── Test/
    ├── TestDbContext.cs               # DbContext
    ├── Entities/
    │   ├── UsersEntity.cs
    │   ├── ProductsEntity.cs
    │   └── OrdersEntity.cs
    └── Configurations/
        ├── UsersEntityConfiguration.cs
        ├── ProductsEntityConfiguration.cs
        └── OrdersEntityConfiguration.cs
```

Use the generated code in your application:

```csharp
using Generated;
using Microsoft.EntityFrameworkCore;

var options = new DbContextOptionsBuilder<TestDbContext>()
    .UseSqlServer("your-connection-string")
    .Options;

using var context = new TestDbContext(options);
var users = await context.Users.ToListAsync();
```

## Advanced Features

### Extend DbContext

Since the generated DbContext is a `partial class`, you can extend it in another file:

```csharp
namespace Generated
{
    public partial class TestDbContext
    {
        partial void OnModelCreatingPartial(ModelBuilder modelBuilder)
        {
            // Add custom configuration
        }
    }
}
```

### Schema Caching

The `.schema` file is cached for performance:
- **First run:** Connects to database and extracts schema
- **Subsequent runs:** Reads from cache file

When your database structure changes, delete the `.schema` file to regenerate:

```bash
rm Generated/*
dotnet build
```

## Supported Databases

- ✅ SQL Server
- ✅ MySQL / MariaDB
- 🚧 PostgreSQL (planned)

## Cross-Platform Support

This package uses `Microsoft.Data.SqlClient` for SQL Server connectivity, which is fully supported on:
- ✅ Windows
- ✅ macOS
- ✅ Linux

## License
MIT License

