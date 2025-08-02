namespace T1.SqlSchemaExtract;

public class FieldSchema
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public int DataSize { get; set; }
    public int DataScale { get; set; }
    public bool IsNullable { get; set; }
    public bool IsPrimaryKey { get; set; }
    public bool IsIdentity { get; set; }
    public string DefaultValue { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
}