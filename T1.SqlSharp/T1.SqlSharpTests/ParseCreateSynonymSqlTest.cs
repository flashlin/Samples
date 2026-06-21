using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateSynonymSqlTest
{
    [Test]
    public void Create_synonym()
    {
        var sql = "CREATE SYNONYM dbo.Orders FOR RemoteServer.Sales.dbo.Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSynonymStatement
        {
            SynonymName = "dbo.Orders",
            ForName = "RemoteServer.Sales.dbo.Orders"
        });
    }

    [Test]
    public void Drop_synonym()
    {
        var sql = "DROP SYNONYM dbo.Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Synonym,
            Names = ["dbo.Orders"]
        });
    }
}
