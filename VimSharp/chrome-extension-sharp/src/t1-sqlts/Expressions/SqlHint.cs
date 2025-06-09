namespace T1.SqlSharp.Expressions;

public class SqlHint : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Hint;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_Hint(this);
    }

    public string Name { get; set; } = string.Empty;
    public string ToSql()
    {
        return Name;
    }
}