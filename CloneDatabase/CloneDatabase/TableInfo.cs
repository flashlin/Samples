namespace CloneDatabase;

public class TableInfo
{
    public string Name { get; set; } = "";
    public List<TableFieldInfo> Fields { get; set; } = new();
}