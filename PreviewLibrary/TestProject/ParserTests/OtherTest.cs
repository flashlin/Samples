using Xunit;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public class OtherTest : ParserTestBase
	{
		public OtherTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void multiComment()
		{
			var sql = "/* 123 */";
			Parse(sql);
			ThenExprShouldBe("/* 123 */");
		}
	}
}
