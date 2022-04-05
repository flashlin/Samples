using Xunit;
using Xunit.Abstractions;

namespace TestProject.ParserTests
{
	public class SelectTest : ParserTestBase
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

		[Fact]
		public void select_field_from_table()
		{
			var sql = "select name from customer";
			Parse(sql);
			ThenExprShouldBe("SELECT name FROM customer");
		}
	}
}
