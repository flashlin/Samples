namespace T1.SqlSharp.Expressions;

public class SqlAlterRoleStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterRoleStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterRoleStatement(this);
    }

    public string RoleName { get; set; } = string.Empty;
    public bool IsAddMember { get; set; }
    public string MemberName { get; set; } = string.Empty;
    public string NewName { get; set; } = string.Empty;

    public string ToSql()
    {
        if (!string.IsNullOrEmpty(NewName))
        {
            return $"ALTER ROLE {RoleName} WITH NAME = {NewName}";
        }

        var action = IsAddMember ? "ADD" : "DROP";
        return $"ALTER ROLE {RoleName} {action} MEMBER {MemberName}";
    }
}
