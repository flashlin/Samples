using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlOptionClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.OptionClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OptionClause(this);
    }

    public List<SqlQueryHint> Hints { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("OPTION (");
        sql.Write(string.Join(", ", Hints.Select(x => x.ToSql())));
        sql.Write(")");
        return sql.ToString();
    }
}
