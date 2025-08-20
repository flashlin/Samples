namespace T1.SqlSchemaExtract;

public class StoreProcedureInfo
{
    public string Name { get; set; } = string.Empty;
    public string Body { get; set; } = string.Empty;
    public List<SynonymInfo> Synonyms { get; set; } = new();
}
