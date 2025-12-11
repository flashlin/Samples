using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.Helper;

namespace T1.SqlSharpTests;

[TestFixture]
public class SqlUpdateExpressionBuilderTest
{
    [Test]
    public void Update_Set_Should_Create_UpdateStatement_With_Correct_SetColumn()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Update")
            .Options;

        using var db = new TestDbContext(options);
        var name = "John";

        var result = SqlUpdateExpressionBuilder.Update(db.Users)
            .Set(u => u.Name, name)
            .Build();

        result.ShouldBe(new SqlUpdateStatement
        {
            TableName = "[dbo].[Users]",
            SetColumns =
            [
                new SqlSetColumn
                {
                    ColumnName = "Name",
                    ParameterName = "@p0",
                    Value = "John"
                }
            ]
        });
    }

    [Test]
    public void Update_Set_Should_Generate_Correct_SQL_String()
    {
        var options = new DbContextOptionsBuilder<TestDbContext>()
            .UseInMemoryDatabase(databaseName: "TestDb_Update_Sql")
            .Options;

        using var db = new TestDbContext(options);
        var name = "John";

        var result = SqlUpdateExpressionBuilder.Update(db.Users)
            .Set(u => u.Name, name)
            .Build();

        var sql = result.ToSql();

        var expectedSql = "UPDATE [dbo].[Users] SET [Name] = @p0";

        sql.Should().Be(expectedSql);
    }
}

