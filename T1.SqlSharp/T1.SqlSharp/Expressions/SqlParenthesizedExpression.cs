namespace T1.SqlSharp.Expressions;

public class SqlParenthesizedExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.Group;
    public required ISqlExpression Inner { get; set; }
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_Group(this);
    }

    public string ToSql()
    {
        return $"({Inner.ToSql()})";
    }

}