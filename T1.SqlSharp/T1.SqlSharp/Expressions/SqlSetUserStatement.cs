namespace T1.SqlSharp.Expressions;

public class SqlSetUserStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.SetUserStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SetUserStatement(this);
    }

    public string UserName { get; set; } = string.Empty;

    public string ToSql()
    {
        return string.IsNullOrEmpty(UserName) ? "SETUSER" : $"SETUSER {UserName}";
    }
}
