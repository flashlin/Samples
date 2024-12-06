namespace T1.SqlSharp.Expressions;

public class SqlHint : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Hint;
    public string Name { get; set; } = string.Empty;
    public string ToSql()
    {
        return Name;
    }
}