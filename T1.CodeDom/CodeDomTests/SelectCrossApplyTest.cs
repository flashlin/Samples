using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class SelectCrossApplyTest : TestBase
{
    public SelectCrossApplyTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void cross_apply()
    {
        var sql = @"select 1, t.name from customer
cross apply (select name from otherTable) as t (name)";
        Parse(sql);
			
        ThenExprShouldBe(@"SELECT 1, t.name FROM customer
CROSS APPLY ( SELECT name FROM otherTable ) AS t(name)");
    }
}