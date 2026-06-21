using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseDdlDropTruncateTest
{
    [Test]
    public void Truncate_table()
    {
        var sql = "TRUNCATE TABLE Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlTruncateTableStatement
        {
            TableName = "Users"
        });
    }

    [Test]
    public void Drop_table_single()
    {
        var sql = "DROP TABLE Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Table,
            Names = ["Users"]
        });
    }

    [Test]
    public void Drop_table_if_exists_multiple()
    {
        var sql = "DROP TABLE IF EXISTS dbo.A, dbo.B";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Table,
            IfExists = true,
            Names = ["dbo.A", "dbo.B"]
        });
    }

    [Test]
    public void Drop_view()
    {
        var sql = "DROP VIEW vCustomer";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.View,
            Names = ["vCustomer"]
        });
    }

    [Test]
    public void Drop_procedure()
    {
        var sql = "DROP PROCEDURE IF EXISTS usp_GetUser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Procedure,
            IfExists = true,
            Names = ["usp_GetUser"]
        });
    }
}
