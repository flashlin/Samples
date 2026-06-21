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
    public bool IsOn { get; set; }

    public string ToSql()
    {
        var state = IsOn ? "ON" : "OFF";
        return string.IsNullOrEmpty(Target)
            ? $"SET {Option} {state}"
            : $"SET {Option} {Target} {state}";
    }
}
