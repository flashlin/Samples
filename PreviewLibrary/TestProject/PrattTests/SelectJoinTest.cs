using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class SelectJoinTest : TestBase
	{
		public SelectJoinTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void left_join()
		{
			var sql = "select 1 from customer left join secondTable";
			Parse(sql);
			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN secondTable");
		}
	}
}
