using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlOrderByClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.OrderByClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OrderByClause(this);
    }

    public List<SqlOrderColumn> Columns { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.WriteLine("ORDER BY");
        sql.Indent++;
        for (var i = 0; i < Columns.Count; i++)
        {
            sql.Write(Columns[i].ToSql());
            if (i < Columns.Count - 1)
            {
                sql.Write(",");
            }
            sql.WriteLine();
        }
        sql.Indent--;
        return sql.ToString();
    }
}