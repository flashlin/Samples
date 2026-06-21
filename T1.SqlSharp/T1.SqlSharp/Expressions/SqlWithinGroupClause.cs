using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlWithinGroupClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WithinGroupClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WithinGroupClause(this);
    }

    public List<SqlOrderColumn> Columns { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("WITHIN GROUP (ORDER BY ");
        sql.Write(string.Join(", ", Columns.Select(x => x.ToSql())));
        sql.Write(")");
        return sql.ToString();
    }
}
