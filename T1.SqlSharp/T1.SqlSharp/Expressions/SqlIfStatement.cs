using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlIfStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.IfStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_IfStatement(this);
    }

    public required ISqlExpression Condition { get; set; }
    public required ISqlExpression Then { get; set; }
    public ISqlExpression? Else { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"IF {Condition.ToSql()} {Then.ToSql()}");
        if (Else != null)
        {
            sql.Append($" ELSE {Else.ToSql()}");
        }
        return sql.ToString();
    }
}
