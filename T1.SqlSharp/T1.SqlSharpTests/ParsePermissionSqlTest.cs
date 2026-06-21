using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParsePermissionSqlTest
{
    [Test]
    public void Grant_select_on_table_to_user()
    {
        var sql = "GRANT SELECT ON Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["SELECT"],
            ObjectName = "Orders",
            Principals = ["appuser"]
        });
    }

    [Test]
    public void Grant_multiple_permissions_with_grant_option()
    {
        var sql = "GRANT SELECT, INSERT ON dbo.Orders TO role1, role2 WITH GRANT OPTION";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["SELECT", "INSERT"],
            ObjectName = "dbo.Orders",
            Principals = ["role1", "role2"],
            WithGrantOption = true
        });
    }

    [Test]
    public void Revoke_execute_from_user()
    {
        var sql = "REVOKE EXECUTE ON usp_DoWork FROM appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Revoke,
            Permissions = ["EXECUTE"],
            ObjectName = "usp_DoWork",
            Principals = ["appuser"]
        });
    }

    [Test]
    public void Deny_update_to_user_cascade()
    {
        var sql = "DENY UPDATE ON Orders TO appuser CASCADE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Deny,
            Permissions = ["UPDATE"],
            ObjectName = "Orders",
            Principals = ["appuser"],
            Cascade = true
        });
    }

    [Test]
    public void Grant_statement_permission_without_object()
    {
        var sql = "GRANT EXECUTE TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["EXECUTE"],
            Principals = ["appuser"]
        });
    }
}
