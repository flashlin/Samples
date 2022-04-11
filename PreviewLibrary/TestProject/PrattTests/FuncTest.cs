using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class FuncTest : TestBase
	{
		public FuncTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void exists()
		{
			var sql = @"exists(1)";
			Parse(sql);

			ThenExprShouldBe(@"EXISTS( 1 )");
		}
	}
}
