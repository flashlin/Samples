namespace T1.SqlSharp.Expressions;

public class SqlCreateRuleOrDefaultStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateRuleOrDefaultStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateRuleOrDefaultStatement(this);
    }

    public bool IsRule { get; set; }
    public string Name { get; set; } = string.Empty;
    public ISqlExpression? Expression { get; set; }

    public string ToSql()
    {
        var kind = IsRule ? "RULE" : "DEFAULT";
        return $"CREATE {kind} {Name} AS {Expression?.ToSql()}";
    }
}
