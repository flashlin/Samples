namespace SqlBoyLib;

public class SqlQuery
{
    public string Statement { get; set; } = string.Empty;
    public string ParameterDefinitions { get; set; } = string.Empty;
    public Dictionary<string, object?> Parameters { get; set; } = new();

    public string ToExecuteSql()
    {
        var paramValues = string.Join(", ", Parameters.Select(p => $"{p.Key} = {FormatValue(p.Value)}"));
        
        if (string.IsNullOrEmpty(ParameterDefinitions))
        {
            return $"EXEC sys.sp_executesql\n  @stmt = N'{Statement}'";
        }

        return $"EXEC sys.sp_executesql\n  @stmt = N'{Statement}',\n  @params = N'{ParameterDefinitions}',\n  {paramValues}";
    }

    private string FormatValue(object? value)
    {
        if (value == null)
            return "NULL";
        
        if (value is string || value is char)
            return $"N'{value}'";
        
        if (value is bool b)
            return b ? "1" : "0";
        
        return value.ToString() ?? "NULL";
    }
}

