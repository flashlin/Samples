using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlQueryHint : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.QueryHint;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_QueryHint(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<ISqlExpression> Arguments { get; set; } = [];
    public bool ArgumentsInParentheses { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Name);
        if (Arguments.Count > 0)
        {
            var joined = string.Join(", ", Arguments.Select(x => x.ToSql()));
            sql.Write(ArgumentsInParentheses ? $" ({joined})" : $" {joined}");
        }
        return sql.ToString();
    }
}
