namespace T1.SqlSharp.Expressions;

public class SqlAlterAuthorizationStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterAuthorizationStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterAuthorizationStatement(this);
    }

    public string SecurableClass { get; set; } = string.Empty;
    public string ObjectName { get; set; } = string.Empty;
    public string Principal { get; set; } = string.Empty;

    public string ToSql()
    {
        var securable = string.IsNullOrEmpty(SecurableClass) ? ObjectName : $"{SecurableClass}::{ObjectName}";
        return $"ALTER AUTHORIZATION ON {securable} TO {Principal}";
    }
}
