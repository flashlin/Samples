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
}
