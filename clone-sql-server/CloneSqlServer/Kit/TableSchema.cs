namespace CloneSqlServer.Kit;

public class TableSchema
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public List<FieldSchema> Fields { get; set; } = [];
}