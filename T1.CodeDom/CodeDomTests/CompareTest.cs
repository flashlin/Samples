using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
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
