namespace T1.SqlSharp.Expressions;

public class SqlTextPointerStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.TextPointerStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TextPointerStatement(this);
    }

    public string Operation { get; set; } = string.Empty;
    public List<string> Arguments { get; set; } = [];

    public string ToSql()
    {
        return $"{Operation} {string.Join(" ", Arguments)}";
    }
}
