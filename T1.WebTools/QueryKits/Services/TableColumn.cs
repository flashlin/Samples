namespace QueryKits.Services;

public class TableColumn
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public int Size { get; set; }
    public int Precision { get; set; }
    public int Scale { get; set; }
}