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

The tool automatically extracts foreign key relationships from your database schema. You can also (optionally) define additional relationships or override automatically extracted ones using Mermaid ER diagram syntax inside the `.db` file.

**Example:**

```
Server=localhost;Database=MyDb;User Id=sa;Password=***;TrustServerCertificate=True
# Table relationships (optional, using Mermaid ER diagram syntax)
User ||--o{ Order : "User.Id = Order.UserId"
User ||--|| Profile : "User.Id = Profile.UserId"
```

**Relationship Syntax (Mermaid ER Diagram):**
- `||--o{` : One-to-Many, Bidirectional (e.g., User has many Orders, Order belongs to User)
- `||-->o{` : One-to-Many, Unidirectional (only principal has navigation property)
- `||--||` : One-to-One, Bidirectional (e.g., User has one Profile, Profile belongs to User)
- `||-->||` : One-to-One, Unidirectional
- `||--o|` : One-to-Zero-or-One, Bidirectional (dependent is optional)
- `o|--||` : Zero-or-One-to-One, Bidirectional (principal is optional)

**Relationship Extraction:**
The tool automatically extracts foreign key relationships from the database schema. Relationships defined in the `.db` file (Mermaid syntax) will **override** automatically extracted ones, allowing you to:
- Specify unidirectional relationships (database FK cannot determine this)
- Customize navigation property names
- Define logical relationships without actual FK constraints


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

### Custom Entity Configuration

All generated EntityConfiguration classes are `partial class` and provide a `ConfigureCustomProperties` partial method for adding custom configurations without modifying the auto-generated code.

**Example:** Create a custom configuration file (e.g., `UsersEntityConfiguration.Custom.cs`):

```csharp
namespace Generated.Databases.MyDb.Configurations
{
    public partial class UsersEntityConfiguration
    {
        partial void ConfigureCustomProperties(EntityTypeBuilder<UsersEntity> builder)
        {
            // Add custom indexes
            builder.HasIndex(x => x.Email)
                .IsUnique()
                .HasDatabaseName("UX_Users_Email");
            
            // Add column comments
            builder.Property(x => x.Email)
                .HasComment("User email address");
            
            // Add custom validations
            builder.Property(x => x.Username)
                .HasMaxLength(50);
        }
    }
}
```

**Benefits:**
- ✅ Custom configuration files are NOT overwritten during regeneration
- ✅ Zero performance overhead (compiler removes unused partial methods)
- ✅ Add indexes, comments, validations, and other EF Core configurations
- ✅ Type-safe with full IntelliSense support
- ✅ Navigation properties and relationships are automatically generated from database FK constraints or Mermaid definitions

**Note:** Avoid overriding auto-generated property configurations in the partial method as this may cause conflicts.

### File Management

**Regeneration Behavior:**
- When running `dotnet build`, the tool checks if files already exist
- If a file already exists, it will be **skipped** (not overwritten)
- If you want to update auto-generated files (e.g., after database schema changes), manually delete the corresponding files or the entire `Generated/` directory

**Example: Updating Generated Code**

```bash
# Option 1: Delete the entire Generated directory
rm -rf Generated/
dotnet build

# Option 2: Delete only specific database files
rm -rf Generated/MyDatabase/
dotnet build

# Option 3: Delete only schema cache file (will reconnect to database)
rm Generated/*.schema
dotnet build
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

