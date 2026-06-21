namespace T1.SqlSharp.Expressions;

public class SqlGotoStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.GotoStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_GotoStatement(this);
    }

    public string Label { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"GOTO {Label}";
    }
}
