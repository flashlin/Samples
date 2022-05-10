using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
	public class WhileTest : TestBase
	{
		public WhileTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void while_name()
		{
			var sql = @"while name=1
begin
	select 1
end";
			Parse(sql);

			ThenExprShouldBe(@"WHILE name = 1
BEGIN
	SELECT 1
END");
		}
	}
}
