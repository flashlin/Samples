namespace T1.SqlSharp.Expressions;

public class SqlWhileStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.WhileStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WhileStatement(this);
    }

    public required ISqlExpression Condition { get; set; }
    public required ISqlExpression Body { get; set; }

    public string ToSql()
    {
        return $"WHILE {Condition.ToSql()} {Body.ToSql()}";
    }
}
