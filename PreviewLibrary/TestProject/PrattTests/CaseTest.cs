using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class CaseTest : TestBase
	{
		public CaseTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void case_when()
		{
			var sql = "case when @id = -1 then [Name] else @Name end";
			Parse(sql);
			ThenExprShouldBe("CASE WHEN @id = -1 THEN [Name] ELSE @Name END");
		}
	}
}
