namespace T1.SqlSharp.Expressions;

public class SqlAlterServerConfigurationStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterServerConfigurationStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterServerConfigurationStatement(this);
    }

    public string Setting { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"ALTER SERVER CONFIGURATION SET {Setting}";
    }
}
