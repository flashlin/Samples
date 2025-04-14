namespace CloneSqlServer;

public class TableIndexSchema
{
    public string TableName { get; set; } = string.Empty;
    public string IndexName { get; set; } = string.Empty;
    public string IndexType { get; set; } = string.Empty;  // PK, FK, INDEX
    public bool IsPrimaryKey { get; set; }
    public bool IsUnique { get; set; }
    public bool IsClustered { get; set; }
    public string ColumnsString { get; set; } = string.Empty;
    public List<string> Columns => string.IsNullOrEmpty(ColumnsString) ? [] : ColumnsString.Split(',').ToList();
    public string ReferencedTableName { get; set; } = string.Empty;
    public string ReferencedColumnsString { get; set; } = string.Empty;
    public List<string> ReferencedColumns => string.IsNullOrEmpty(ReferencedColumnsString) ? [] : ReferencedColumnsString.Split(',').ToList();
}