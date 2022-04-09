using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class SetTest : TestBase
	{
		public SetTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void set_ANSI_NULLS()
		{
			var sql = "set ansi_nulls off";
			Parse(sql);
			ThenExprShouldBe("SET ANSI_NULLS OFF");
		}
	}
}
