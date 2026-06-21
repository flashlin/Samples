using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterAuthorizationSqlTest
{
    [Test]
    public void Alter_authorization_on_object()
    {
        var sql = "ALTER AUTHORIZATION ON OBJECT::dbo.Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterAuthorizationStatement
        {
            SecurableClass = "OBJECT",
            ObjectName = "dbo.Orders",
            Principal = "appuser"
        });
    }

    [Test]
    public void Alter_authorization_without_class()
    {
        var sql = "ALTER AUTHORIZATION ON Orders TO appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterAuthorizationStatement
        {
            ObjectName = "Orders",
            Principal = "appuser"
        });
    }
}
