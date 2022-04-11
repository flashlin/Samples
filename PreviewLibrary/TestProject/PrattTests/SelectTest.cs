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
		public void select_name_number()
		{
			var sql = "select name, 1";
			Parse(sql);
			ThenExprShouldBe("SELECT name, 1");
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

		[Fact]
		public void select_number_from_dbo_table()
		{
			var sql = @"select 1 from dbo.customer";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer");
		}

		[Fact]
		public void select_number_from_dbo_table_where()
		{
			var sql = @"select 1 from dbo.customer where name=customFunc()";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE name = customFunc()");
		}
	}
}
