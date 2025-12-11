using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Helper;

namespace T1.SqlSharpTests;

[TestFixture]
public class SqlInsertExpressionBuilderTest
{
    [Test]
    public void Into_DbSet_Should_Create_InsertStatement_With_Correct_TableName()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Insert")
            .Options;

        using var db = new TestDbContext(options);

        var result = SqlInsertExpressionBuilder.Into(db.Users).Build();

        result.ShouldBe(new SqlInsertStatement
        {
            TableName = "[dbo].[Users]",
            Columns = ["Id", "Name", "Birth"]
        });
    }

    [Test]
    public void Into_DbSet_Should_Generate_Correct_SQL_String()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Insert_Sql")
            .Options;

        using var db = new TestDbContext(options);

        var result = SqlInsertExpressionBuilder.Into(db.Users).Build();

        var sql = result.ToSql();

        var expectedSql = "INSERT INTO [dbo].[Users] ([Id], [Name], [Birth]) VALUES (@p0, @p1, @p2)";

        sql.Should().Be(expectedSql);
    }
}

