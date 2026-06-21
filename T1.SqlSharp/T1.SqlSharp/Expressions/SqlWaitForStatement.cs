namespace T1.SqlSharp.Expressions;

public enum SqlWaitForKind
{
    Delay,
    Time
}

public class SqlWaitForStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.WaitForStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WaitForStatement(this);
    }

    public SqlWaitForKind Kind { get; set; }
    public required ISqlExpression Time { get; set; }

    public string ToSql()
    {
        var keyword = Kind == SqlWaitForKind.Delay ? "DELAY" : "TIME";
        return $"WAITFOR {keyword} {Time.ToSql()}";
    }
}
