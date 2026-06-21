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
    public void Grant_with_securable_class_prefix()
    {
        var sql = "GRANT SELECT ON OBJECT::dbo.Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["SELECT"],
            SecurableClass = "OBJECT",
            ObjectName = "dbo.Orders",
            Principals = ["appuser"]
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

    [Test]
    public void Grant_multiword_permission_on_object()
    {
        var sql = "GRANT VIEW DEFINITION ON dbo.Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["VIEW DEFINITION"],
            ObjectName = "dbo.Orders",
            Principals = ["appuser"]
        });
    }

    [Test]
    public void Grant_multiword_statement_permission()
    {
        var sql = "GRANT CREATE TABLE TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["CREATE TABLE"],
            Principals = ["appuser"]
        });
    }

    [Test]
    public void Revoke_grant_option_for()
    {
        var sql = "REVOKE GRANT OPTION FOR SELECT ON Orders FROM appuser CASCADE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Revoke,
            GrantOptionFor = true,
            Permissions = ["SELECT"],
            ObjectName = "Orders",
            Principals = ["appuser"],
            Cascade = true
        });
    }

    [Test]
    public void Grant_with_as_grantor()
    {
        var sql = "GRANT SELECT ON Orders TO appuser AS dbo";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["SELECT"],
            ObjectName = "Orders",
            Principals = ["appuser"],
            AsGrantor = "dbo"
        });
    }

    [Test]
    public void Grant_column_level()
    {
        var sql = "GRANT SELECT (col1, col2) ON dbo.Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["SELECT"],
            Columns = ["col1", "col2"],
            ObjectName = "dbo.Orders",
            Principals = ["appuser"]
        });
    }

    [Test]
    public void Grant_mixed_single_and_multiword_permissions()
    {
        var sql = "GRANT SELECT, VIEW DEFINITION ON dbo.Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlPermissionStatement
        {
            Action = SqlPermissionAction.Grant,
            Permissions = ["SELECT", "VIEW DEFINITION"],
            ObjectName = "dbo.Orders",
            Principals = ["appuser"]
        });
    }
}
