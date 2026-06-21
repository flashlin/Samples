using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreatePrincipalSqlTest
{
    [Test]
    public void Create_role_with_authorization()
    {
        var sql = "CREATE ROLE SalesRole AUTHORIZATION dbo";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreatePrincipalStatement
        {
            Kind = SqlPrincipalKind.Role,
            Name = "SalesRole",
            Authorization = "dbo"
        });
    }

    [Test]
    public void Create_user_for_login()
    {
        var sql = "CREATE USER AppUser FOR LOGIN AppLogin";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreatePrincipalStatement
        {
            Kind = SqlPrincipalKind.User,
            Name = "AppUser",
            ForLogin = "AppLogin"
        });
    }

    [Test]
    public void Create_login_with_password()
    {
        var sql = "CREATE LOGIN AppLogin WITH PASSWORD = 'P@ss'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreatePrincipalStatement
        {
            Kind = SqlPrincipalKind.Login,
            Name = "AppLogin",
            Password = "'P@ss'"
        });
    }
}
