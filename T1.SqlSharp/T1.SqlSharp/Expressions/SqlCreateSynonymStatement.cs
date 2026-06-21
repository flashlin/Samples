namespace T1.SqlSharp.Expressions;

public class SqlCreateSynonymStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateSynonymStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateSynonymStatement(this);
    }

    public string SynonymName { get; set; } = string.Empty;
    public string ForName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"CREATE SYNONYM {SynonymName} FOR {ForName}";
    }
}
