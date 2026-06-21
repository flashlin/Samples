using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlBlockStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.BlockStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_BlockStatement(this);
    }

    public List<ISqlExpression> Statements { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("BEGIN ");
        sql.Append(string.Join("; ", Statements.Select(s => s.ToSql())));
        sql.Append(" END");
        return sql.ToString();
    }
}
