using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlWindowClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WindowClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WindowClause(this);
    }

    public List<SqlWindowDefinition> Definitions { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("WINDOW ");
        sql.Write(string.Join(", ", Definitions.Select(x => x.ToSql())));
        return sql.ToString();
    }
}
