namespace CloneSqlServer;

public class TableIndexSchema
{
    public string TableName { get; set; } = string.Empty;
    public string IndexName { get; set; } = string.Empty;
    public string IndexType { get; set; } = string.Empty;  // PK, FK, INDEX
    public bool IsPrimaryKey { get; set; }
    public bool IsUnique { get; set; }
    public bool IsClustered { get; set; }
    public List<string> Columns { get; set; } = new();
    public string? ReferencedTableName { get; set; }  // 只有 FK 才會有值
    public List<string> ReferencedColumns { get; set; } = new();  // 只有 FK 才會有值
    public string ColumnsString { get; set; }
}