namespace T1.SqlSharp.Expressions;

public enum SqlLoopControlAction
{
    Break,
    Continue
}

public class SqlLoopControlStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.LoopControlStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_LoopControlStatement(this);
    }

    public SqlLoopControlAction Action { get; set; }

    public string ToSql()
    {
        return Action == SqlLoopControlAction.Break ? "BREAK" : "CONTINUE";
    }
}
