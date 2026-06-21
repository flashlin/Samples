using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlCollateExpression : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.CollateExpression;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CollateExpression(this);
    }

    public required ISqlExpression Expression { get; set; }
    public string Collation { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Expression.ToSql());
        sql.Write($" COLLATE {Collation}");
        return sql.ToString();
    }
}
