using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlAlterObjectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterObjectStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterObjectStatement(this);
    }

    public string Kind { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"ALTER {Kind}");
        if (!string.IsNullOrEmpty(Name))
        {
            sql.Append($" {Name}");
        }

        if (!string.IsNullOrEmpty(Action))
        {
            sql.Append($" {Action}");
        }

        return sql.ToString();
    }
}
