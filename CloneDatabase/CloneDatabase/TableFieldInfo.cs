namespace CloneDatabase;

public class TableFieldInfo
{
    public string Name { get; set; }
    public string DataType { get; set; }
    public int MaxLength { get; set; }
    public int Precision { get; set; }
    public int Scale { get; set; }
    public bool IsNullable { get; set; }
    public bool IsPrimaryKey { get; set; }
}