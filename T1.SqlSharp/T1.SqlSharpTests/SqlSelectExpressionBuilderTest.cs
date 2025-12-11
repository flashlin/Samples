using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Helper;

namespace T1.SqlSharpTests;

[TestFixture]
public class SqlSelectExpressionBuilderTest
{
    [Test]
    public void From_DbSet_Should_Create_SelectStatement_With_Correct_TableSource()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb")
            .Options;

        using var db = new TestDbContext(options);

        var result = SqlSelectExpressionBuilder.From(db.Users).Build();

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
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]", Withs = [new SqlHint { Name = "NOLOCK" }] }]
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

        var result = SqlSelectExpressionBuilder.From(db.Users)
            .Where(u => u.Name == userName)
            .Build();

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
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]", Withs = [new SqlHint { Name = "NOLOCK" }] }],
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

        var result = SqlSelectExpressionBuilder.From(db.Users)
            .Where(u => u.Name == userName)
            .Build();

        var sql = result.ToSql();

        var expectedSql = "SELECT\n\t[dbo].[Users].[Id] AS Users_Id,\n\t[dbo].[Users].[Name] AS Users_Name,\n\t[dbo].[Users].[Birth] AS Users_Birth\nFROM \n\t[dbo].[Users] WITH(NOLOCK)\nWHERE \n\t[dbo].[Users].[Name] = @p0\n";

        sql.Should().Be(expectedSql);
    }

    [Test]
    public void Select_SingleColumn_Distinct_Should_Generate_Correct_SelectStatement()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Distinct")
            .Options;
        using var db = new TestDbContext(options);

        var result = SqlSelectExpressionBuilder.From(db.Users)
            .Select(u => u.Name)
            .Distinct()
            .Build();

        result.ShouldBe(new SelectStatement
        {
            SelectType = SelectType.Distinct,
            Columns =
            [
                new SelectColumn
                {
                    Field = new SqlColumnExpression
                    {
                        Schema = "dbo",
                        TableName = "Users",
                        ColumnName = "Name"
                    },
                    Alias = "Users_Name"
                }
            ],
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]", Withs = [new SqlHint { Name = "NOLOCK" }] }]
        });
    }

    [Test]
    public void Select_SingleColumn_Distinct_Should_Generate_Correct_SQL_String()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Distinct_Sql")
            .Options;
        using var db = new TestDbContext(options);

        var result = SqlSelectExpressionBuilder.From(db.Users)
            .Select(u => u.Name)
            .Distinct()
            .Build();

        var sql = result.ToSql();

        var expectedSql = "SELECT DISTINCT \n\t[dbo].[Users].[Name] AS Users_Name\nFROM \n\t[dbo].[Users] WITH(NOLOCK)\n";

        sql.Should().Be(expectedSql);
    }

    [Test]
    public void Where_With_Take_Should_Generate_Correct_SelectStatement()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Take")
            .Options;
        using var db = new TestDbContext(options);
        var id = 1;

        var result = SqlSelectExpressionBuilder.From(db.Users)
            .Where(u => u.Id == id)
            .Take(1)
            .Build();

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
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]", Withs = [new SqlHint { Name = "NOLOCK" }] }],
            Where = new SqlConditionExpression
            {
                Left = new SqlColumnExpression
                {
                    Schema = "dbo",
                    TableName = "Users",
                    ColumnName = "Id"
                },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlParameter
                {
                    ParameterName = "@p0",
                    Value = 1
                }
            },
            Top = new SqlTopClause
            {
                Expression = new SqlValue
                {
                    SqlType = SqlType.IntValue,
                    Value = "1"
                }
            }
        });
    }

    [Test]
    public void Where_With_Take_Should_Generate_Correct_SQL_String()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Take_Sql")
            .Options;
        using var db = new TestDbContext(options);
        var id = 1;

        var result = SqlSelectExpressionBuilder.From(db.Users)
            .Where(u => u.Id == id)
            .Take(1)
            .Build();

        var sql = result.ToSql();

        var expectedSql = "SELECT\nTOP 1\n\t[dbo].[Users].[Id] AS Users_Id,\n\t[dbo].[Users].[Name] AS Users_Name,\n\t[dbo].[Users].[Birth] AS Users_Birth\nFROM \n\t[dbo].[Users] WITH(NOLOCK)\nWHERE \n\t[dbo].[Users].[Id] = @p0\n";

        sql.Should().Be(expectedSql);
    }
}

