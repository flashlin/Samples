namespace T1.SqlSharp.Expressions;

public class SqlAsExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AsExpr;
    public required ISqlExpression Instance { get; set; }
    public required ISqlExpression As { get; set; }

    public string ToSql()
    {
        return $"{Instance.ToSql()} as {As.ToSql()}";
    }
}

public class SqlAliasExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AliasExpr;
    public required string Name { get; set; } = string.Empty;
    public string ToSql()
    {
        return $"AS {Name}";
    }
}