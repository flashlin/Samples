namespace T1.SqlSchemaExtract;

public class UniqueTableInfo
{
    public string TableName { get; set; } = string.Empty;
    public string ConstraintName { get; set; } = string.Empty;
    public List<string> Fields { get; set; } = new();
}
