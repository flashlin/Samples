# Entity Class 設計規範

## EntityConfiguration 配置規範

每個 Entity 都必須配置對應的 EntityConfiguration，並遵循以下規則：

### 1. 資料表設定
使用 `ToTable` 明確指定資料表名稱

### 2. 主鍵設定
使用 `HasKey` 設定主鍵，通常為 `Id`

### 3. Id 自動流水號
主鍵 Id 必須設定為自動產生：
- 使用 `ValueGeneratedOnAdd()` 設定自動遞增

### 4. 屬性型別宣告
所有屬性都必須明確宣告資料庫型別：
- 使用 `HasColumnType` 指定資料庫型別（例如 `varchar(50)`, `int`, `datetime` 等）
- 字串型別使用 `HasMaxLength` 限制長度
- 必填欄位使用 `IsRequired()`

### 範例程式碼

```csharp
public class QueryPanelConfiguration : IEntityTypeConfiguration<QueryPanel>
{
    public void Configure(EntityTypeBuilder<QueryPanel> builder)
    {
        builder.ToTable("_QueryPanel");

        builder.HasKey(e => e.Id);

        builder.Property(e => e.Id)
            .ValueGeneratedOnAdd();

        builder.Property(e => e.TableName)
            .HasMaxLength(50)
            .HasColumnType("varchar(50)")
            .IsRequired();

        builder.Property(e => e.Description)
            .HasMaxLength(200)
            .HasColumnType("nvarchar(200)");

        builder.Property(e => e.CreatedAt)
            .HasColumnType("datetime")
            .IsRequired();
    }
}
```

### 重點檢查清單
- ✅ 必須有 `HasColumnType` 宣告資料庫類型
- ✅ Id 必須設定為自動流水號 `ValueGeneratedOnAdd()`
- ✅ 字串欄位要同時設定 `HasMaxLength` 和 `HasColumnType`
- ✅ 根據需求設定 `IsRequired()`
