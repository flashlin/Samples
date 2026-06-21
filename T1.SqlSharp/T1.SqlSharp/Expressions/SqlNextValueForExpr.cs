namespace T1.SqlSharp.Expressions;

public class SqlNextValueForExpr : ISqlExpression
{
    public SqlType SqlType => SqlType.NextValueForExpr;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_NextValueForExpr(this);
    }

    public string SequenceName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"NEXT VALUE FOR {SequenceName}";
    }
}
