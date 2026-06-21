using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterPrincipalSqlTest
{
    [Test]
    public void Alter_role_add_member()
    {
        var sql = "ALTER ROLE db_datareader ADD MEMBER appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterRoleStatement
        {
            RoleName = "db_datareader",
            IsAddMember = true,
            MemberName = "appuser"
        });
    }

    [Test]
    public void Alter_role_drop_member()
    {
        var sql = "ALTER ROLE db_datareader DROP MEMBER appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterRoleStatement
        {
            RoleName = "db_datareader",
            IsAddMember = false,
            MemberName = "appuser"
        });
    }

    [Test]
    public void Alter_role_rename()
    {
        var sql = "ALTER ROLE sales WITH NAME = sales_v2";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterRoleStatement
        {
            RoleName = "sales",
            NewName = "sales_v2"
        });
    }

    [Test]
    public void Alter_login_disable()
    {
        var sql = "ALTER LOGIN appuser DISABLE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterPrincipalStatement
        {
            Kind = SqlPrincipalKind.Login,
            Name = "appuser",
            Action = "DISABLE"
        });
    }

    [Test]
    public void Alter_user_with_default_schema()
    {
        var sql = "ALTER USER appuser WITH DEFAULT_SCHEMA = dbo";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterPrincipalStatement
        {
            Kind = SqlPrincipalKind.User,
            Name = "appuser",
            Options = ["DEFAULT_SCHEMA = dbo"]
        });
    }

    [Test]
    public void Alter_login_with_password()
    {
        var sql = "ALTER LOGIN appuser WITH PASSWORD = 'newpass'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterPrincipalStatement
        {
            Kind = SqlPrincipalKind.Login,
            Name = "appuser",
            Options = ["PASSWORD = 'newpass'"]
        });
    }
}
