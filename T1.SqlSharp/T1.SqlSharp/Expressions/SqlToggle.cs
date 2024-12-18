namespace T1.SqlSharp.Expressions;

public class SqlToggle : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WithToggle;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_Toggle(this);
    }

    public string ToggleName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;


    public string ToSql()
    {
        return $"{ToggleName}={Value}";
    }
}