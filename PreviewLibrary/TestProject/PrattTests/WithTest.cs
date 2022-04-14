using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class WithTest : TestBase
	{
		public WithTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void with_table()
		{
			var sql = @"with tb1(id,name)
as (
	select id, name from customer
)";
			Parse(sql);
			ThenExprShouldBe(@"WITH tb1(id, name)
AS (
	SELECT id, name
	FROM customer
)");
		}
	}
}
