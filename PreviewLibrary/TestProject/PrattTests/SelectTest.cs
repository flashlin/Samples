using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class SelectTest : TestBase
	{
		public SelectTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void select_number()
		{
			var sql = "select 1";
			Parse(sql);
			ThenExprShouldBe("SELECT 1");
		}
	}
}
