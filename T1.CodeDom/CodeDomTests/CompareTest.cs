using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class CompareTest : TestBase
	{
		public CompareTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void in_number()
		{
			var sql = @"a in (1,2)";

			Parse(sql);

			ThenExprShouldBe(@"a IN (1, 2)");
		}
	}
}
