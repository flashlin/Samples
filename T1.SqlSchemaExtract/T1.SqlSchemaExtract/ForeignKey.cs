namespace T1.SqlSchemaExtract;

public class ForeignKey
{
    public string DefineName { get; set; } = string.Empty;
    public string ForeignTableName { get; set; } = string.Empty;
    public string PrimaryTableName { get; set; } = string.Empty;
    public List<string> ForeignKeyNames { get; set; } = [];
    public List<string> PrimaryKeyNames { get; set; } = [];
}