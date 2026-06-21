namespace T1.SqlSharp.Expressions;

public enum SqlWaitForKind
{
    Delay,
    Time,
    Receive
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
    public ISqlExpression? Timeout { get; set; }

    public string ToSql()
    {
        if (Kind == SqlWaitForKind.Receive)
        {
            var timeout = Timeout == null ? string.Empty : $", TIMEOUT {Timeout.ToSql()}";
            return $"WAITFOR ({Time.ToSql()}){timeout}";
        }

        var keyword = Kind == SqlWaitForKind.Delay ? "DELAY" : "TIME";
        return $"WAITFOR {keyword} {Time.ToSql()}";
    }
}
