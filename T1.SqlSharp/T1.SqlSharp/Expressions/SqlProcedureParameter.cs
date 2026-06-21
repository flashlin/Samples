namespace T1.SqlSharp.Expressions;

public class SqlProcedureParameter
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public SqlDataSize? DataSize { get; set; }
    public ISqlExpression? DefaultValue { get; set; }
    public bool IsOutput { get; set; }
    public bool IsReadOnly { get; set; }
}
