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

		[Fact]
		public void select_name()
		{
			var sql = "select name";
			Parse(sql);
			ThenExprShouldBe("SELECT name");
		}

		[Fact]
		public void select_name_without_as_name()
		{
			var sql = "select customerName name";
			Parse(sql);
			ThenExprShouldBe("SELECT customerName AS name");
		}

		[Fact]
		public void select_name_as_name()
		{
			var sql = "select customerName as name";
			Parse(sql);
			ThenExprShouldBe("SELECT customerName AS name");
		}
	}
}
