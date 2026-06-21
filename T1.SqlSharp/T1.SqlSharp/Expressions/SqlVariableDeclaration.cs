namespace T1.SqlSharp.Expressions;

public class SqlVariableDeclaration
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public SqlDataSize? DataSize { get; set; }
    public ISqlExpression? InitialValue { get; set; }
}
