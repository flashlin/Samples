using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlWindowFrameClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WindowFrameClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WindowFrameClause(this);
    }

    public SqlFrameUnit Unit { get; set; }
    public SqlWindowFrameBound Start { get; set; } = new();
    public SqlWindowFrameBound? End { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Unit.ToString().ToUpper());
        sql.Write(" ");
        if (End != null)
        {
            sql.Write($"BETWEEN {Start.ToSql()} AND {End.ToSql()}");
        }
        else
        {
            sql.Write(Start.ToSql());
        }
        return sql.ToString();
    }
}
