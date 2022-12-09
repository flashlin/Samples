namespace CloneDatabase;

public class TableIndexInfo
{
    public string Name { get; set; }
    public string ColumnName { get; set; }
    public int IndexColumnId { get; set; }
    public int KeyOrdinal { get; set; }
    public bool IsIncludedColumn { get; set; }
}