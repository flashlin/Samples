using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests;

public class OverTest : TestBase
{
    public OverTest(ITestOutputHelper outputHelper) : base(outputHelper)
    {
    }

    [Fact]
    public void over_partition()
    {
        var sql = "over(partition by name)";
        Parse(sql);
        ThenExprShouldBe(@"OVER( PARTITION BY name )");
    }
    
    [Fact]
    public void over_order_by()
    {
        var sql = "over(order by name asc)";
        Parse(sql);
        ThenExprShouldBe(@"OVER( ORDER BY name ASC )");
    }
}