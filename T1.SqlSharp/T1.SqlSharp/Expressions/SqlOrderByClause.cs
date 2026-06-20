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
    public ISqlExpression? Offset { get; set; }
    public ISqlExpression? Fetch { get; set; }

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
        WriteOffsetFetchSql(sql);
        return sql.ToString();
    }

    private void WriteOffsetFetchSql(IndentStringBuilder sql)
    {
        if (Offset == null)
        {
            return;
        }

        sql.WriteLine($"OFFSET {Offset.ToSql()} ROWS");
        if (Fetch != null)
        {
            sql.WriteLine($"FETCH NEXT {Fetch.ToSql()} ROWS ONLY");
        }
    }
}