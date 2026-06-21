using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateSchemaDatabaseSqlTest
{
    [Test]
    public void Create_schema()
    {
        var sql = "CREATE SCHEMA Sales";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSchemaStatement { SchemaName = "Sales" });
    }

    [Test]
    public void Create_schema_with_authorization()
    {
        var sql = "CREATE SCHEMA Sales AUTHORIZATION dbo";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSchemaStatement { SchemaName = "Sales", Authorization = "dbo" });
    }

    [Test]
    public void Create_database()
    {
        var sql = "CREATE DATABASE ShopDb";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement { DatabaseName = "ShopDb" });
    }
}
