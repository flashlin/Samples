namespace T1.SqlSharp.Expressions;

public class SqlCreateMessageTypeStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateMessageTypeStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateMessageTypeStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string Validation { get; set; } = string.Empty;

    public string ToSql()
    {
        var validation = string.IsNullOrEmpty(Validation) ? string.Empty : $" VALIDATION = {Validation}";
        return $"CREATE MESSAGE TYPE {Name}{validation}";
    }
}
