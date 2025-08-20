namespace T1.SqlSchemaExtract;

public class TableConstraintInfo
{
    public string TableName { get; set; } = string.Empty;
    public string ConstraintName { get; set; } = string.Empty;
    public List<string> FieldNames { get; set; } = new();
}
