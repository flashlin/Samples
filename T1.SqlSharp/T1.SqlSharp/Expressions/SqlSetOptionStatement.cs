namespace T1.SqlSharp.Expressions;

public class SqlSetOptionStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.SetOptionStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SetOptionStatement(this);
    }

    public string Option { get; set; } = string.Empty;
    public string Target { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return string.IsNullOrEmpty(Target)
            ? $"SET {Option} {Value}"
            : $"SET {Option} {Target} {Value}";
    }
}
