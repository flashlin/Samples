# T1.EfCodeFirstGenerator 使用指南

## 概述

T1.EfCodeFirstGenerator 是一個兩階段的程式碼生成工具，用於從資料庫 schema 自動產生 Entity Framework Core Code First 程式碼。

## 架構

### 1. CLI 工具 (T1.EfCodeFirstGenerator.CLI)
- 連接到資料庫
- 提取 schema 資訊
- 產生 `.schema` 檔案（JSON 格式）

### 2. Source Generator (T1.EfCodeFirstGenerator)
- 在編譯時讀取 `.schema` 檔案
- 自動產生 EF Core 程式碼
- 程式碼直接加入編譯（不寫入硬碟）

## 完整使用流程

### 步驟 1: 準備連線字串檔案

在專案目錄建立 `.db` 檔案（例如：`databases.db`）：

```
# SQL Server 連線
Server=localhost;Database=MyDatabase;User Id=sa;Password=YourPassword;TrustServerCertificate=true

# MySQL 連線
Server=192.168.1.100;Database=AnotherDb;Uid=root;Pwd=secret;
```

**注意事項：**
- 每一行一個連線字串
- 以 `#` 或 `//` 開頭的行視為註解
- 支援標準 ADO.NET 連線字串格式

### 步驟 2: 執行 CLI 工具提取 Schema

```bash
# 方法 1: 直接執行
cd YourProjectDirectory
dotnet run --project ../T1.EfCodeFirstGenerator.CLI/T1.EfCodeFirstGenerator.CLI.csproj

# 方法 2: 指定目錄
dotnet run --project ../T1.EfCodeFirstGenerator.CLI/T1.EfCodeFirstGenerator.CLI.csproj -- /path/to/your/project
```

執行後會產生 `{ServerName}_{DatabaseName}.schema` 檔案，例如：
- `localhost_MyDatabase.schema`
- `192.168.1.100_AnotherDb.schema`

### 步驟 3: 設定專案引用

在您的專案 `.csproj` 檔案中加入：

```xml
<ItemGroup>
  <!-- 引用 Source Generator -->
  <ProjectReference Include="../T1.EfCodeFirstGenerator/T1.EfCodeFirstGenerator.csproj" 
                    OutputItemType="Analyzer" 
                    ReferenceOutputAssembly="false" />
</ItemGroup>

<ItemGroup>
  <!-- 將 .schema 檔案標記為 AdditionalFiles -->
  <AdditionalFiles Include="*.schema" />
</ItemGroup>

<ItemGroup>
  <!-- 加入必要的 EF Core 套件 -->
  <PackageReference Include="Microsoft.EntityFrameworkCore" Version="8.0.0" />
  <PackageReference Include="Microsoft.EntityFrameworkCore.SqlServer" Version="8.0.0" />
  <!-- 或 MySQL -->
  <!-- <PackageReference Include="Pomelo.EntityFrameworkCore.MySql" Version="8.0.0" /> -->
</ItemGroup>
```

### 步驟 4: 建置專案

```bash
dotnet build
```

建置時，Source Generator 會自動產生以下程式碼：

1. **DbContext**: `{DatabaseName}DbContext.cs`
2. **Entity 類別**: `{TableName}Entity.cs` （每個資料表一個）
3. **Entity Configuration**: `{TableName}EntityConfiguration.cs` （每個資料表一個）

### 步驟 5: 使用生成的程式碼

```csharp
using Generated.Example; // namespace 根據 .schema 檔案位置而定

// 使用 DbContext
public class MyService
{
    private readonly SampleDbDbContext _context;

    public MyService(SampleDbDbContext context)
    {
        _context = context;
    }

    public async Task<List<UsersEntity>> GetAllUsers()
    {
        return await _context.Users.ToListAsync();
    }

    public async Task AddUser(UsersEntity user)
    {
        _context.Users.Add(user);
        await _context.SaveChangesAsync();
    }
}
```

## 生成的程式碼範例

假設有一個 `Users` 資料表：

### UsersEntity.cs
```csharp
using System;

namespace Generated.Example
{
    public class UsersEntity
    {
        public int Id { get; set; }
        public required string Username { get; set; }
        public required string Email { get; set; }
        public DateTime CreatedAt { get; set; }
        public bool IsActive { get; set; }
    }
}
```

### UsersEntityConfiguration.cs
```csharp
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated.Example
{
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

            builder.Property(x => x.Username)
                .HasColumnType("nvarchar(100)")
                .IsRequired()
                .HasMaxLength(100);

            builder.Property(x => x.Email)
                .HasColumnType("nvarchar(255)")
                .IsRequired()
                .HasMaxLength(255);

            builder.Property(x => x.CreatedAt)
                .HasColumnType("datetime2")
                .IsRequired()
                .HasDefaultValue(getdate());

            builder.Property(x => x.IsActive)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true);
        }
    }
}
```

### SampleDbDbContext.cs
```csharp
using Microsoft.EntityFrameworkCore;

namespace Generated.Example
{
    public partial class SampleDbDbContext : DbContext
    {
        public DbSet<UsersEntity> Users { get; set; }
        public DbSet<ProductsEntity> Products { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.ApplyConfiguration(new UsersEntityConfiguration());
            modelBuilder.ApplyConfiguration(new ProductsEntityConfiguration());
        }
    }
}
```

## Schema 檔案格式

`.schema` 檔案是 JSON 格式：

```json
{
  "DatabaseName": "SampleDb",
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
        },
        {
          "FieldName": "Username",
          "SqlDataType": "nvarchar(100)",
          "IsPrimaryKey": false,
          "IsNullable": false,
          "DefaultValue": null
        }
      ]
    }
  ]
}
```

## 支援的資料庫

- ✅ SQL Server
- ✅ MySQL / MariaDB
- 🚧 PostgreSQL (規劃中)
- 🚧 Oracle (規劃中)

## SQL 型別對應

| SQL Type | C# Type |
|----------|---------|
| int, integer | int |
| bigint | long |
| smallint | short |
| tinyint | byte |
| bit, boolean | bool |
| decimal, numeric, money | decimal |
| float, real | double |
| date, datetime, datetime2 | DateTime |
| time | TimeSpan |
| uniqueidentifier, guid | Guid |
| varchar, nvarchar, text | string |
| binary, varbinary, image | byte[] |

## 常見問題

### Q1: 如何更新 Schema？
刪除 `.schema` 檔案，重新執行 CLI 工具。

### Q2: 如何自訂型別對應？
可以修改 `SqlTypeToCSharpTypeConverter` 類別並註冊自訂對應規則。

### Q3: 生成的程式碼在哪裡？
程式碼在編譯時產生並直接加入記憶體，不會寫入硬碟。可以在 IDE 中透過 "Go to Definition" 查看。

### Q4: 如何自訂 namespace？
Namespace 根據 `.schema` 檔案所在的目錄決定。

### Q5: 支援複合主鍵嗎？
是的，會自動偵測並產生對應的 `HasKey` 設定。

## 進階用法

### 自訂 DbContext

由於生成的 DbContext 是 `partial class`，您可以在另一個檔案中擴展它：

```csharp
// SampleDbDbContext.Extensions.cs
namespace Generated.Example
{
    public partial class SampleDbDbContext
    {
        public SampleDbDbContext(DbContextOptions<SampleDbDbContext> options)
            : base(options)
        {
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!optionsBuilder.IsConfigured)
            {
                optionsBuilder.UseSqlServer("your-connection-string");
            }
        }
    }
}
```

### 註冊到 DI 容器

```csharp
// Program.cs or Startup.cs
services.AddDbContext<SampleDbDbContext>(options =>
    options.UseSqlServer(Configuration.GetConnectionString("DefaultConnection")));
```

## 最佳實踐

1. **版本控制**: 將 `.schema` 檔案加入版本控制，但不要提交 `.db` 檔案（包含密碼）
2. **環境分離**: 不同環境使用不同的 `.db` 檔案
3. **定期更新**: 當資料庫 schema 變更時，重新產生 `.schema` 檔案
4. **部分類別**: 使用 `partial class` 擴展生成的程式碼，不要直接修改生成的檔案

## 疑難排解

### 錯誤: "Could not load file or assembly 'Newtonsoft.Json'"
確保 Source Generator 專案正確配置了 Newtonsoft.Json 套件。

### 錯誤: "No schema files found"
確保 `.schema` 檔案已加入 `<AdditionalFiles>` 中。

### 錯誤: "Connection failed"
檢查連線字串是否正確，資料庫是否可存取。

