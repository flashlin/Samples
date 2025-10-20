# T1.EfCodeFirstGenerator 詳細使用指南

## 目錄

1. [安裝方式](#安裝方式)
2. [基本使用](#基本使用)
3. [進階配置](#進階配置)
4. [MSBuild 整合](#msbuild-整合)
5. [生成程式碼說明](#生成程式碼說明)
6. [自訂擴展](#自訂擴展)
7. [疑難排解](#疑難排解)

## 安裝方式

### 方式 1: CLI 工具（開發環境）

適用於開發階段，手動控制程式碼生成時機。

```bash
# Clone 或下載專案
git clone https://github.com/your-repo/T1.EfCodeFirstGenerator.git

# 建置專案
cd T1.EfCodeFirstGenerator/T1.EfCodeFirstGenerator
dotnet build

# 在目標專案執行
cd /path/to/your/project
dotnet run --project /path/to/T1.EfCodeFirstGenerator/T1.EfCodeFirstGenerator.csproj -- .
```

### 方式 2: NuGet 套件（生產環境）

適用於團隊協作和 CI/CD 環境。

```bash
# 安裝套件
dotnet add package T1.EfCodeFirstGenerator

# 建置時自動執行
dotnet build
```

## 基本使用

### 步驟 1: 準備連線字串檔案

在專案根目錄建立 `.db` 檔案（例如：`databases.db`）：

```
# 註解行以 # 或 // 開頭

# SQL Server 範例
Server=localhost;Database=MyDatabase;User Id=sa;Password=YourPassword;TrustServerCertificate=true

# MySQL 範例  
Server=192.168.1.100;Database=TestDb;Uid=root;Pwd=secret;

# 多個資料庫
Server=localhost;Database=DB1;User Id=sa;Password=pass1;
Server=localhost;Database=DB2;User Id=sa;Password=pass2;
```

**支援的連線字串格式：**
- SQL Server: 標準 ADO.NET 格式
- MySQL: MySQL Connector/NET 格式

### 步驟 2: 執行程式碼生成

**使用 CLI：**
```bash
dotnet run --project path/to/T1.EfCodeFirstGenerator.csproj -- /path/to/your/project
```

**使用 NuGet（自動）：**
```bash
dotnet build  # MSBuild Task 會自動執行
```

### 步驟 3: 檢視生成的程式碼

程式碼會生成在 `Generated/` 目錄：

```
Generated/
├── localhost_MyDatabase.schema          # Schema 快取檔案
├── MyDatabaseDbContext.cs              # DbContext
├── Entities/
│   ├── UsersEntity.cs
│   ├── ProductsEntity.cs
│   └── OrdersEntity.cs
└── Configurations/
    ├── UsersEntityConfiguration.cs
    ├── ProductsEntityConfiguration.cs
    └── OrdersEntityConfiguration.cs
```

### 步驟 4: 使用生成的程式碼

```csharp
using Generated;
using Microsoft.EntityFrameworkCore;

public class Program
{
    public static async Task Main(string[] args)
    {
        // 配置 DbContext
        var options = new DbContextOptionsBuilder<MyDatabaseDbContext>()
            .UseSqlServer("Server=localhost;Database=MyDatabase;...")
            .Options;

        using var context = new MyDatabaseDbContext(options);

        // CRUD 操作
        var users = await context.Users.ToListAsync();
        
        var newUser = new UsersEntity
        {
            Username = "john_doe",
            Email = "john@example.com",
            CreatedAt = DateTime.Now,
            IsActive = true
        };

        context.Users.Add(newUser);
        await context.SaveChangesAsync();
    }
}
```

## 進階配置

### 自訂 DbContext

由於生成的 DbContext 是 `partial class`，您可以在另一個檔案中擴展：

```csharp
// MyDatabaseDbContext.Extensions.cs
using Microsoft.EntityFrameworkCore;

namespace Generated
{
    public partial class MyDatabaseDbContext
    {
        // 自訂建構子
        public MyDatabaseDbContext(DbContextOptions<MyDatabaseDbContext> options)
            : base(options)
        {
        }

        // 自訂配置
        partial void OnModelCreatingPartial(ModelBuilder modelBuilder)
        {
            // 添加自訂配置
            modelBuilder.Entity<UsersEntity>()
                .HasMany(u => u.Orders)
                .WithOne(o => o.User);
        }
    }
}
```

### 排除特定資料表

編輯 `.schema` 檔案，移除不需要的資料表：

```json
{
  "DatabaseName": "MyDatabase",
  "Tables": [
    {
      "TableName": "Users",
      "Fields": [...]
    }
    // 移除不需要的資料表
  ]
}
```

### 自訂 Namespace

修改 `Program.cs` 中的 `targetNamespace`：

```csharp
var targetNamespace = "YourCompany.Data.Models";
```

### 自訂型別對應

```csharp
var converter = new SqlTypeToCSharpTypeConverter();

// 註冊自訂對應
converter.RegisterCustomMapping("geometry", (sqlType, isNullable) => 
    isNullable ? "NetTopologySuite.Geometries.Geometry?" : "NetTopologySuite.Geometries.Geometry");

var generator = new EfCodeGenerator(converter);
```

## MSBuild 整合

### NuGet 套件如何工作

當您安裝 `T1.EfCodeFirstGenerator` NuGet 套件時：

1. `build/T1.EfCodeFirstGenerator.targets` 會被導入到專案
2. `GenerateEfCodeTask` 在 `BeforeBuild` 目標執行
3. Task 掃描 `.db` 檔案並產生程式碼
4. 生成的 `.cs` 檔案被包含在編譯中

### 自訂 MSBuild 行為

在您的 `.csproj` 中：

```xml
<PropertyGroup>
  <!-- 停用自動生成 -->
  <T1SkipCodeGeneration>true</T1SkipCodeGeneration>
</PropertyGroup>
```

### 手動觸發生成

```bash
dotnet msbuild /t:T1GenerateEfCode
```

## 生成程式碼說明

### DbContext

```csharp
public partial class MyDatabaseDbContext : DbContext
{
    public DbSet<UsersEntity> Users { get; set; }
    public DbSet<ProductsEntity> Products { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.ApplyConfiguration(new UsersEntityConfiguration());
        modelBuilder.ApplyConfiguration(new ProductsEntityConfiguration());
    }
}
```

### Entity

```csharp
public class UsersEntity
{
    public int Id { get; set; }
    public required string Username { get; set; }
    public required string Email { get; set; }
    public DateTime CreatedAt { get; set; }
    public bool IsActive { get; set; }
}
```

**注意：** 非 nullable 的 reference type（如 `string`）會加上 `required` 修飾符。

### EntityConfiguration

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

        builder.Property(x => x.Username)
            .HasColumnType("nvarchar(100)")
            .IsRequired()
            .HasMaxLength(100);

        builder.Property(x => x.CreatedAt)
            .HasColumnType("datetime2")
            .IsRequired()
            .HasDefaultValue(getdate());
    }
}
```

## 自訂擴展

### 添加導航屬性

```csharp
// UsersEntity.Extensions.cs
namespace Generated
{
    public partial class UsersEntity
    {
        public virtual ICollection<OrdersEntity> Orders { get; set; }
    }
}

// UsersEntityConfiguration.Extensions.cs
namespace Generated
{
    public partial class UsersEntityConfiguration
    {
        partial void ConfigureRelations(EntityTypeBuilder<UsersEntity> builder)
        {
            builder.HasMany(u => u.Orders)
                .WithOne(o => o.User)
                .HasForeignKey(o => o.UserId);
        }
    }
}
```

### 添加資料驗證

```csharp
using System.ComponentModel.DataAnnotations;

namespace Generated
{
    public partial class UsersEntity : IValidatableObject
    {
        public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
        {
            if (string.IsNullOrWhiteSpace(Username))
            {
                yield return new ValidationResult(
                    "Username is required",
                    new[] { nameof(Username) });
            }
        }
    }
}
```

## 疑難排解

### 問題 1: "No .db files found"

**原因：** 專案目錄沒有 `.db` 檔案。

**解決：** 確認 `.db` 檔案存在於專案根目錄。

### 問題 2: "Login failed for user"

**原因：** 資料庫連線字串不正確或資料庫無法存取。

**解決：** 
- 檢查連線字串是否正確
- 確認資料庫服務正在執行
- 檢查防火牆設定

### 問題 3: 生成的程式碼無法編譯

**原因：** 可能是 schema 檔案損壞或型別對應問題。

**解決：**
- 刪除 `.schema` 檔案重新產生
- 檢查自訂型別對應是否正確

### 問題 4: MSBuild Task 不執行

**原因：** NuGet 套件未正確安裝或 targets 檔案未被導入。

**解決：**
```bash
dotnet restore
dotnet clean
dotnet build
```

### 問題 5: "Duplicate Compile items"

**原因：** 手動添加了 `<Compile Include="Generated\**\*.cs" />`。

**解決：** 移除手動添加的 Compile 項目，SDK 會自動包含。

## Git 版本控制最佳實踐

### 建議 Commit 的檔案

```gitignore
# Commit schema 檔案（小且穩定）
Generated/*.schema

# Commit 生成的程式碼（方便 code review）
Generated/**/*.cs
```

### 不建議 Commit 的檔案

```gitignore
# 不要 commit .db 檔案（包含敏感資訊）
*.db

# 使用環境變數或加密存儲
# 在 CI/CD 中動態生成 .db 檔案
```

### CI/CD 整合

```yaml
# .github/workflows/build.yml
steps:
  - name: Setup .db file
    run: |
      echo "Server=${{ secrets.DB_SERVER }};Database=${{ secrets.DB_NAME }};..." > databases.db
  
  - name: Generate code
    run: dotnet run --project T1.EfCodeFirstGenerator -- .
  
  - name: Build
    run: dotnet build
```

## 效能考量

### Schema 快取

`.schema` 檔案會被快取，避免重複連接資料庫：

- **首次執行：** 連接資料庫，提取 schema，產生 `.schema` 檔案
- **後續執行：** 直接讀取 `.schema` 檔案，快速產生程式碼

### 何時重新生成 Schema

- 資料庫結構變更（新增/修改資料表）
- 需要更新欄位屬性（型別、長度、預設值）

```bash
# 強制重新生成
rm Generated/*.schema
dotnet run --project T1.EfCodeFirstGenerator -- .
```

## 相關資源

- [Entity Framework Core 文件](https://docs.microsoft.com/ef/core/)
- [Fluent API 參考](https://docs.microsoft.com/ef/core/modeling/)
- [MSBuild Tasks](https://docs.microsoft.com/visualstudio/msbuild/msbuild-tasks)

## 支援

如有問題或建議，請：
- 提交 GitHub Issue
- 查看常見問題
- 參考範例專案
