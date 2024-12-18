using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlHavingClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.HavingClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_HavingClause(this);
    }

    public ISqlExpression Condition { get; set; } = new SqlValue();

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("HAVING ");
        sql.Write(Condition.ToSql());
        return sql.ToString();
    }
}