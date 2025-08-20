namespace T1.SqlSchemaExtract;

public class ClusteredIndexInfo
{
    public string TableName { get; set; } = string.Empty;
    public string IndexName { get; set; } = string.Empty;
    public List<FieldIndexInfo> IndexFields { get; set; } = new();
}
