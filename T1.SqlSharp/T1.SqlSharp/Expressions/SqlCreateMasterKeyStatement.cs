namespace T1.SqlSharp.Expressions;

public class SqlCreateMasterKeyStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateMasterKeyStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateMasterKeyStatement(this);
    }

    public string Password { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"CREATE MASTER KEY ENCRYPTION BY PASSWORD = {Password}";
    }
}
