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
    public void Drop_login()
    {
        var sql = "DROP LOGIN appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Login,
            Names = ["appuser"]
        });
    }

    [Test]
    public void Drop_user_if_exists()
    {
        var sql = "DROP USER IF EXISTS appuser";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.User,
            IfExists = true,
            Names = ["appuser"]
        });
    }

    [Test]
    public void Drop_role()
    {
        var sql = "DROP ROLE app_role";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Role,
            Names = ["app_role"]
        });
    }

    [Test]
    public void Drop_index_on_table()
    {
        var sql = "DROP INDEX ix_Name ON Customer";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Index,
            Names = ["ix_Name"],
            OnTable = "Customer"
        });
    }

    [Test]
    public void Drop_index_if_exists_on_table()
    {
        var sql = "DROP INDEX IF EXISTS ix_Name ON Customer";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlDropStatement
        {
            ObjectType = SqlDropObjectType.Index,
            IfExists = true,
            Names = ["ix_Name"],
            OnTable = "Customer"
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
