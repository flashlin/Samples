namespace T1.SqlSharp.Expressions;

public class SqlVariableDeclaration
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public SqlDataSize? DataSize { get; set; }
    public ISqlExpression? InitialValue { get; set; }
    public bool IsTable { get; set; }
    public List<SqlColumnDefinition> TableColumns { get; set; } = [];
    public List<ISqlConstraint> TableConstraints { get; set; } = [];
    public bool IsCursor { get; set; }
    public ISqlExpression? CursorSource { get; set; }
    public List<string> CursorOptions { get; set; } = [];
}
