namespace T1.SqlSharp.Expressions;

public class SqlUseStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.UseStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_UseStatement(this);
    }

    public string DatabaseName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"USE {DatabaseName}";
    }
}
