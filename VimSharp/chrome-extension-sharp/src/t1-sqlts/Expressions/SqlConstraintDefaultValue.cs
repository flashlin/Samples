namespace T1.SqlSharp.Expressions;

public class SqlConstraintDefaultValue : ISqlConstraint
{
    public SqlType SqlType { get; } = SqlType.ConstraintDefaultValue;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ConstraintDefaultValue(this);
    }

    public string ConstraintName { get; set; } = string.Empty;
    public string DefaultValue { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"DEFAULT {DefaultValue}";
    }
}