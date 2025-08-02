namespace T1.SqlSchemaExtract;

public class ForeignKeyInfo
{
    public string DefineName { get; set; } = string.Empty;
    public string ForeignTableName { get; set; } = string.Empty;
    public string ForeignKeyName { get; set; } = string.Empty;
    public string PrimaryTableName { get; set; } = string.Empty;
    public string PrimaryKeyName { get; set; } = string.Empty;
}