using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlWithCte : ISqlExpression
{
    public SqlType SqlType => SqlType.WithCte;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WithCte(this);
    }

    public List<SqlCommonTableExpression> CommonTableExpressions { get; set; } = [];
    public required ISqlExpression Statement { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("WITH ");
        for (var i = 0; i < CommonTableExpressions.Count; i++)
        {
            sql.Write(CommonTableExpressions[i].ToSql());
            if (i < CommonTableExpressions.Count - 1)
            {
                sql.Write(", ");
            }
        }
        sql.WriteLine();
        sql.Write(Statement.ToSql());
        return sql.ToString();
    }
}
