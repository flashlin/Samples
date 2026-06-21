namespace T1.SqlSharp.Expressions;

public class SqlGoStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.GoStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_GoStatement(this);
    }

    public int? Count { get; set; }

    public string ToSql()
    {
        return Count.HasValue ? $"GO {Count.Value}" : "GO";
    }
}
