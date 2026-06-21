using System.Text;

namespace T1.SqlSharp.Expressions;

public enum SqlPrincipalKind
{
    Login,
    User,
    Role
}

public class SqlCreatePrincipalStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreatePrincipalStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreatePrincipalStatement(this);
    }

    public SqlPrincipalKind Kind { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Authorization { get; set; } = string.Empty;
    public string ForLogin { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE {Kind.ToString().ToUpperInvariant()} {Name}");
        if (!string.IsNullOrEmpty(Authorization))
        {
            sql.Append($" AUTHORIZATION {Authorization}");
        }

        if (!string.IsNullOrEmpty(ForLogin))
        {
            sql.Append($" FOR LOGIN {ForLogin}");
        }

        if (!string.IsNullOrEmpty(Password))
        {
            sql.Append($" WITH PASSWORD = {Password}");
            if (Options.Count > 0)
            {
                sql.Append($", {string.Join(", ", Options)}");
            }
        }
        else if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
