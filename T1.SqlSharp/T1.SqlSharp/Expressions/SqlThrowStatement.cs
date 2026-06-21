using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlThrowStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.ThrowStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ThrowStatement(this);
    }

    public ISqlExpression? ErrorNumber { get; set; }
    public ISqlExpression? Message { get; set; }
    public ISqlExpression? State { get; set; }

    public string ToSql()
    {
        if (ErrorNumber == null)
        {
            return "THROW";
        }

        var sql = new StringBuilder();
        sql.Append("THROW ");
        sql.Append(string.Join(", ", new[] { ErrorNumber, Message, State }
            .Where(x => x != null)
            .Select(x => x!.ToSql())));
        return sql.ToString();
    }
}
