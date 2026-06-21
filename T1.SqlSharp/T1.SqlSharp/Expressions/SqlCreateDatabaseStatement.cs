namespace T1.SqlSharp.Expressions;

public class SqlCreateDatabaseStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateDatabaseStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateDatabaseStatement(this);
    }

    public string DatabaseName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"CREATE DATABASE {DatabaseName}";
    }
}
