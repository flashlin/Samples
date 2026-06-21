using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlAlterPrincipalStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterPrincipalStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterPrincipalStatement(this);
    }

    public SqlPrincipalKind Kind { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"ALTER {Kind.ToString().ToUpperInvariant()} {Name}");
        if (!string.IsNullOrEmpty(Action))
        {
            sql.Append($" {Action}");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
