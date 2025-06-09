namespace T1.SqlSharp.Expressions;

public class SqlBetweenValue : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.BetweenValue;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_BetweenValue(this);
    }
    public required ISqlExpression Start { get; set; }
    public required ISqlExpression End { get; set; }
    public string ToSql()
    {
        return $"BETWEEN {Start.ToSql()} AND {End.ToSql()}";
    }
}