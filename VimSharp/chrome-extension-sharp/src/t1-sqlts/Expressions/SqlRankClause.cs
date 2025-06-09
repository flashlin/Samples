using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlRankClause : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.RankClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_RankClause(this);
    }

    public ISqlExpression? PartitionBy { get; set; }
    public required ISqlExpression OrderBy { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("RANK() OVER (");
        if(PartitionBy != null)
        {
            sql.Write(PartitionBy.ToSql());
        }
        sql.Write(" " + OrderBy.ToSql());
        sql.Write(")");
        return sql.ToString();
    }
}