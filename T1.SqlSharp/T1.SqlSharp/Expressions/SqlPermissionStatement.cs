using System.Text;

namespace T1.SqlSharp.Expressions;

public enum SqlPermissionAction
{
    Grant,
    Revoke,
    Deny
}

public class SqlPermissionStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.PermissionStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_PermissionStatement(this);
    }

    public SqlPermissionAction Action { get; set; }
    public List<string> Permissions { get; set; } = [];
    public List<string> Columns { get; set; } = [];
    public string SecurableClass { get; set; } = string.Empty;
    public string ObjectName { get; set; } = string.Empty;
    public List<string> Principals { get; set; } = [];
    public bool GrantOptionFor { get; set; }
    public bool WithGrantOption { get; set; }
    public bool Cascade { get; set; }
    public string AsGrantor { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(ActionToSql());
        sql.Append(' ');
        if (GrantOptionFor)
        {
            sql.Append("GRANT OPTION FOR ");
        }

        sql.Append(string.Join(", ", Permissions));
        if (Columns.Count > 0)
        {
            sql.Append($" ({string.Join(", ", Columns)})");
        }

        if (!string.IsNullOrEmpty(ObjectName))
        {
            var securable = string.IsNullOrEmpty(SecurableClass) ? ObjectName : $"{SecurableClass}::{ObjectName}";
            sql.Append($" ON {securable}");
        }

        sql.Append(Action == SqlPermissionAction.Revoke ? " FROM " : " TO ");
        sql.Append(string.Join(", ", Principals));
        if (WithGrantOption)
        {
            sql.Append(" WITH GRANT OPTION");
        }

        if (Cascade)
        {
            sql.Append(" CASCADE");
        }

        if (!string.IsNullOrEmpty(AsGrantor))
        {
            sql.Append($" AS {AsGrantor}");
        }

        return sql.ToString();
    }

    private string ActionToSql()
    {
        return Action switch
        {
            SqlPermissionAction.Grant => "GRANT",
            SqlPermissionAction.Revoke => "REVOKE",
            SqlPermissionAction.Deny => "DENY",
            _ => string.Empty
        };
    }
}
