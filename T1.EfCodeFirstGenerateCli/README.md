# T1.EfCodeFirstGenerateCli

自動從資料庫 schema 產生 Entity Framework Core Code First 程式碼的工具。

## 特點

- ✅ 支援 SQL Server 和 MySQL
- ✅ 自動產生 DbContext、Entity 和 EntityConfiguration
- ✅ 可作為 CLI 工具或 NuGet 套件使用
- ✅ MSBuild 整合，自動在建置前執行
- ✅ 產生的程式碼包含完整的 Fluent API 配置
- ✅ 支援主鍵、外鍵、索引、預設值等資料庫特性

## 快速開始

### 方法 1: 開發階段（使用 ProjectReference）

**適用於開發和偵錯 T1.EfCodeFirstGenerateCli**

1. **建立連線字串檔案**

在專案目錄建立 `example.db` 檔案：

```
# SQL Server
Server=localhost;Database=MyDatabase;User Id=sa;Password=YourPassword;TrustServerCertificate=true

# MySQL
Server=localhost;Database=TestDb;Uid=root;Pwd=password;
```

2. **設定專案參考**

在 `YourProject.csproj` 中添加：

```xml
<ItemGroup>
  <!-- 開發環境：引用專案以便偵錯 -->
  <ProjectReference Include="../T1.EfCodeFirstGenerateCli/T1.EfCodeFirstGenerateCli.csproj">
    <ReferenceOutputAssembly>false</ReferenceOutputAssembly>
    <OutputItemType>Analyzer</OutputItemType>
  </ProjectReference>
</ItemGroup>

<!-- 導入 MSBuild targets -->
<Import Project="../T1.EfCodeFirstGenerateCli/build/T1.EfCodeFirstGenerateCli.targets" 
        Condition="Exists('../T1.EfCodeFirstGenerateCli/build/T1.EfCodeFirstGenerateCli.targets')" />

<PropertyGroup>
  <!-- 指定 Task Assembly 路徑 -->
  <T1EfCodeFirstGeneratorTaskAssembly>$(MSBuildThisFileDirectory)../T1.EfCodeFirstGenerateCli/bin/$(Configuration)/net8.0/T1.EfCodeFirstGenerateCli.dll</T1EfCodeFirstGeneratorTaskAssembly>
</PropertyGroup>
```

3. **建置專案**

MSBuild Task 會自動執行，產生程式碼到 `Generated/` 目錄：

```bash
dotnet build
```

**優點：**
- ✅ 可以在 T1.EfCodeFirstGenerateCli 中設定中斷點偵錯
- ✅ 修改 CLI 工具後自動重建
- ✅ 建置時自動產生程式碼
- ✅ 建置目標正確（Example 專案的目錄）

### 方法 2: 使用 NuGet 套件（生產環境）

1. **安裝套件**

```bash
dotnet add package T1.EfCodeFirstGenerateCli
```

2. **建立連線字串檔案**

在專案根目錄建立 `.db` 檔案（內容同上）。

3. **建置專案**

MSBuild Task 會自動在建置前執行，產生程式碼到 `Generated/` 目錄。

```bash
dotnet build
```

## 生成的程式碼結構

```
YourProject/
├── databases.db
└── Generated/
    ├── {ServerName}_{DatabaseName}.schema
    ├── {DatabaseName}DbContext.cs
    ├── Entities/
    │   ├── UsersEntity.cs
    │   ├── ProductsEntity.cs
    │   └── ...
    └── Configurations/
        ├── UsersEntityConfiguration.cs
        ├── ProductsEntityConfiguration.cs
        └── ...
```

## 使用範例

```csharp
using Generated;
using Microsoft.EntityFrameworkCore;

// 配置 DbContext
var options = new DbContextOptionsBuilder<SampleDbDbContext>()
    .UseSqlServer("your-connection-string")
    .Options;

using var context = new SampleDbDbContext(options);

// 使用生成的 Entity
var users = await context.Users.ToListAsync();

var newUser = new UsersEntity
{
    Username = "john",
    Email = "john@example.com",
    CreatedAt = DateTime.Now,
    IsActive = true
};

context.Users.Add(newUser);
await context.SaveChangesAsync();
```

## 打包為 NuGet 套件

```bash
cd T1.EfCodeFirstGenerateCli
dotnet pack -c Release

# 測試本地套件
dotnet add package T1.EfCodeFirstGenerateCli --source ./bin/Release
```

## 支援的資料庫

| 資料庫 | 狀態 |
|--------|------|
| SQL Server | ✅ 完全支援 |
| MySQL / MariaDB | ✅ 完全支援 |
| PostgreSQL | 🚧 規劃中 |
| Oracle | 🚧 規劃中 |

## 跨平台支援

此套件使用 `Microsoft.Data.SqlClient` 連接 SQL Server，完全支援：
- ✅ Windows
- ✅ macOS
- ✅ Linux

## SQL 型別對應

| SQL 型別 | C# 型別 |
|----------|---------|
| int, integer | int |
| bigint | long |
| varchar, nvarchar, text | string |
| decimal, numeric | decimal |
| bit, boolean | bool |
| datetime, datetime2 | DateTime |
| uniqueidentifier | Guid |
| binary, varbinary | byte[] |

## 設定選項

### 自訂 Namespace

預設 namespace 為 `Generated`。可以在 CLI 工具中修改 `Program.cs` 來自訂。

### 排除特定資料表

編輯生成的 `.schema` 檔案，移除不需要的資料表。

### 自訂型別對應

修改 `SqlTypeToCSharpTypeConverter` 類別來註冊自訂對應規則。

## Git 版本控制建議

```gitignore
# 建議 commit .schema 檔案（小且穩定）
# *.schema

# 建議 commit Generated/ 目錄（方便查看變更）
# Generated/

# 不要 commit .db 檔案（包含密碼）
*.db
```

## 常見問題

### Q: 如何更新 Schema？

刪除 `.schema` 檔案，重新執行 CLI 工具。

### Q: 可以手動修改生成的程式碼嗎？

可以！由於 DbContext 是 `partial class`，您可以在另一個檔案中擴展它。

### Q: 支援複合主鍵嗎？

是的，會自動偵測並產生對應的 `HasKey` 設定。

### Q: 如何處理多個資料庫？

在 `.db` 檔案中添加多行連線字串，每行一個資料庫。

## License

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！
