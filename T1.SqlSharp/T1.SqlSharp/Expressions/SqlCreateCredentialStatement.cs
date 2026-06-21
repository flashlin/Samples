using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateCredentialStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateCredentialStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateCredentialStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public bool IsDatabaseScoped { get; set; }
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(IsDatabaseScoped ? "CREATE DATABASE SCOPED CREDENTIAL " : "CREATE CREDENTIAL ");
        sql.Append(Name);
        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
