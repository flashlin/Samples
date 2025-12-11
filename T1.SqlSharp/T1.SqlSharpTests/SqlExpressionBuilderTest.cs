using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Helper;

namespace T1.SqlSharpTests;

[TestFixture]
public class SqlExpressionBuilderTest
{
    [Test]
    public void From_DbSet_Should_Create_SelectStatement_With_Correct_TableSource()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb")
            .Options;

        using var db = new TestDbContext(options);

        var result = SqlExpressionBuilder.From(db.Users).Select();

        result.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlColumnExpression { Schema = "dbo", TableName = "Users", ColumnName = "Id" },
                    Alias = "Users_Id"
                },
                new SelectColumn
                {
                    Field = new SqlColumnExpression { Schema = "dbo", TableName = "Users", ColumnName = "Name" },
                    Alias = "Users_Name"
                },
                new SelectColumn
                {
                    Field = new SqlColumnExpression { Schema = "dbo", TableName = "Users", ColumnName = "Birth" },
                    Alias = "Users_Birth"
                }
            ],
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]" }]
        });
    }

    [Test]
    public void Where_SimpleEquality_Should_Generate_Correct_Where_Clause()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb")
            .Options;
        using var db = new TestDbContext(options);
        var userName = "John";

        var result = SqlExpressionBuilder.From(db.Users)
            .Where(u => u.Name == userName)
            .Select();

        result.ShouldBe(new SelectStatement
        {
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlColumnExpression { Schema = "dbo", TableName = "Users", ColumnName = "Id" },
                    Alias = "Users_Id"
                },
                new SelectColumn
                {
                    Field = new SqlColumnExpression { Schema = "dbo", TableName = "Users", ColumnName = "Name" },
                    Alias = "Users_Name"
                },
                new SelectColumn
                {
                    Field = new SqlColumnExpression { Schema = "dbo", TableName = "Users", ColumnName = "Birth" },
                    Alias = "Users_Birth"
                }
            ],
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]" }],
            Where = new SqlConditionExpression
            {
                Left = new SqlColumnExpression
                {
                    Schema = "dbo",
                    TableName = "Users",
                    ColumnName = "Name"
                },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlParameter
                {
                    ParameterName = "@p0",
                    Value = "John"
                }
            }
        });
    }

    [Test]
    public void Where_SimpleEquality_Should_Generate_Correct_SQL_String()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_SqlString")
            .Options;
        using var db = new TestDbContext(options);
        var userName = "John";

        var result = SqlExpressionBuilder.From(db.Users)
            .Where(u => u.Name == userName)
            .Select();

        var sql = result.ToSql();

        var expectedSql = "SELECT\n\t[dbo].[Users].[Id] AS Users_Id,\n\t[dbo].[Users].[Name] AS Users_Name,\n\t[dbo].[Users].[Birth] AS Users_Birth\nFROM \n\t[dbo].[Users]\nWHERE \n\t[dbo].[Users].[Name] = @p0\n";

        sql.Should().Be(expectedSql);
    }
}
