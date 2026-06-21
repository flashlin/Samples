using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateObjectStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateObjectStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateObjectStatement(this);
    }

    public string Kind { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE {Kind}");
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
