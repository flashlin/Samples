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
            FromSources = [new SqlTableSource { TableName = "[dbo].[Users]" }]
        });
    }
}
