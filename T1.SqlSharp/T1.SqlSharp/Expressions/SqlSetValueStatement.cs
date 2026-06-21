namespace T1.SqlSharp.Expressions;

public class SqlSetValueStatement : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.SetValueStatement;
    public required ISqlExpression Name { get; set; } = new SqlValue();
    public string Operator { get; set; } = "=";
    public required ISqlExpression Value { get; set; } = new SqlValue();
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SetValueStatement(this);
    }

    public string ToSql()
    {
        return $"SET {Name.ToSql()} {Operator} {Value.ToSql()}";
    }
}
