using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateSecurityPolicyStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateSecurityPolicyStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateSecurityPolicyStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<string> Predicates { get; set; } = [];
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE SECURITY POLICY {Name}");
        foreach (var predicate in Predicates)
        {
            sql.Append($" ADD {predicate}");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Options)})");
        }

        return sql.ToString();
    }
}
