# T1.EfCodeFirstGenerator

A two-stage C# tool that generates Entity Framework Core Code First entities, configurations, and DbContext from database schemas.

## Architecture

This tool consists of two components:

1. **CLI Tool** (`T1.EfCodeFirstGenerator.CLI`): Connects to databases and extracts schema to `.schema` files
2. **Source Generator** (`T1.EfCodeFirstGenerator`): Reads `.schema` files and generates EF Core code during compilation

## Features

- Extracts database schema from SQL Server and MySQL
- Generates schema files (`.schema`) in JSON format
- Generates EF Core entities with proper property types
- Generates entity configurations with complete Fluent API setup
- Generates DbContext with all DbSets and configurations
- Customizable type mappings

## Installation

### Step 1: Install CLI Tool (Global)

```bash
cd T1.EfCodeFirstGenerator.CLI
dotnet pack
dotnet tool install --global --add-source ./bin/Debug T1.EfCodeFirstGenerator.CLI
```

Or run locally:
```bash
cd T1.EfCodeFirstGenerator.CLI
dotnet run -- <path-to-your-project>
```

### Step 2: Add Source Generator to Your Project

Add this project as an analyzer reference in your target project:

```xml
<ItemGroup>
  <ProjectReference Include="../T1.EfCodeFirstGenerator/T1.EfCodeFirstGenerator.csproj" 
                    OutputItemType="Analyzer" 
                    ReferenceOutputAssembly="false" />
</ItemGroup>
```

## Usage

### 1. Create a `.db` file

Create a file with `.db` extension (e.g., `connections.db`) in your project directory and add connection strings, one per line:

```
Server=localhost;Database=MyDatabase;User Id=sa;Password=MyPassword;TrustServerCertificate=true
Server=192.168.1.100;Database=AnotherDb;Uid=root;Pwd=secret;
```

Lines starting with `#` or `//` are treated as comments.

### 2. Run CLI Tool to Extract Schema

Run the CLI tool in your project directory:

```bash
cd YourProject
dotnet run --project ../T1.EfCodeFirstGenerator.CLI/T1.EfCodeFirstGenerator.CLI.csproj
```

This will:
- Scan for all `.db` files in the directory
- Connect to each database
- Extract table schemas (fields, types, primary keys, nullable, defaults)
- Save to `{ServerName}_{DatabaseName}.schema` files

### 3. Include `.schema` Files as AdditionalFiles

Add to your project's `.csproj`:

```xml
<ItemGroup>
  <AdditionalFiles Include="*.schema" />
</ItemGroup>
```

### 4. Build Your Project

When you build, the Source Generator will:
1. Read all `.schema` files
2. Generate for each schema:
   - `{DatabaseName}DbContext.cs`
   - `{TableName}Entity.cs` for each table
   - `{TableName}EntityConfiguration.cs` for each table

The generated code is added directly to your compilation (in-memory, not written to disk).

## Generated Code Example

For a database with a `Users` table, the generator creates:

**MyDatabaseDbContext.cs**
```csharp
public partial class MyDatabaseDbContext : DbContext
{
    public DbSet<UsersEntity> Users { get; set; }
    
    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.ApplyConfiguration(new UsersEntityConfiguration());
    }
}
```

**UsersEntity.cs**
```csharp
public class UsersEntity
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string Email { get; set; }
}
```

**UsersEntityConfiguration.cs**
```csharp
public class UsersEntityConfiguration : IEntityTypeConfiguration<UsersEntity>
{
    public void Configure(EntityTypeBuilder<UsersEntity> builder)
    {
        builder.ToTable("Users");
        builder.HasKey(x => x.Id);
        
        builder.Property(x => x.Id)
            .HasColumnType("int")
            .ValueGeneratedOnAdd()
            .IsRequired();
        
        builder.Property(x => x.Name)
            .HasColumnType("nvarchar(100)")
            .IsRequired()
            .HasMaxLength(100);
        
        builder.Property(x => x.Email)
            .HasColumnType("nvarchar(255)")
            .IsRequired()
            .HasMaxLength(255);
    }
}
```

## Supported Databases

- SQL Server
- MySQL
- PostgreSQL (planned)
- Oracle (planned)

## Custom Type Mappings

You can extend the type converter by registering custom mappings:

```csharp
var converter = new SqlTypeToCSharpTypeConverter();
converter.RegisterCustomMapping("geometry", (sqlType, isNullable) => 
    isNullable ? "Geometry?" : "Geometry");
```

## Schema File Format

Schema files are JSON format with the following structure:

```json
{
  "DatabaseName": "MyDatabase",
  "Tables": [
    {
      "TableName": "Users",
      "Fields": [
        {
          "FieldName": "Id",
          "SqlDataType": "int",
          "IsPrimaryKey": true,
          "IsNullable": false,
          "DefaultValue": null
        }
      ]
    }
  ]
}
```

## Notes

- Schema files are cached to avoid repeated database connections
- Delete `.schema` files to force schema regeneration
- Connection strings support standard ADO.NET format
- Generated files are placed in memory during compilation (not written to disk)

