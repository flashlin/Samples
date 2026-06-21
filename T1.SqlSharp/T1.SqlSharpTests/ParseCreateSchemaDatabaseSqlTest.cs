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
    public void Create_schema_with_inline_table()
    {
        var sql = "CREATE SCHEMA Sales AUTHORIZATION dbo CREATE TABLE Foo (Id INT)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSchemaStatement
        {
            SchemaName = "Sales",
            Authorization = "dbo",
            Elements =
            [
                new SqlCreateTableExpression
                {
                    TableName = "Foo",
                    Columns =
                    [
                        new SqlColumnDefinition { ColumnName = "Id", DataType = "INT" }
                    ]
                }
            ]
        });
    }

    [Test]
    public void Create_schema_with_inline_table_and_grant()
    {
        var sql = "CREATE SCHEMA Sales CREATE TABLE Foo (Id INT) GRANT SELECT ON Foo TO public";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateSchemaStatement
        {
            SchemaName = "Sales",
            Elements =
            [
                new SqlCreateTableExpression
                {
                    TableName = "Foo",
                    Columns =
                    [
                        new SqlColumnDefinition { ColumnName = "Id", DataType = "INT" }
                    ]
                },
                new SqlPermissionStatement
                {
                    Action = SqlPermissionAction.Grant,
                    Permissions = ["SELECT"],
                    ObjectName = "Foo",
                    Principals = ["public"]
                }
            ]
        });
    }

    [Test]
    public void Create_database()
    {
        var sql = "CREATE DATABASE ShopDb";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateDatabaseStatement { DatabaseName = "ShopDb" });
    }
}
