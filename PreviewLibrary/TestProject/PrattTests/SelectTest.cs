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
		public void select_number_comment_number_from_table()
		{
			var sql = @"select 1, --test
2 from customer";
			Parse(sql);

			ThenExprShouldBe(@"SELECT 1, 2 FROM customer");
		}


		[Fact]
		public void select_name()
		{
			var sql = "select name";
			Parse(sql);
			ThenExprShouldBe("SELECT name");
		}

		[Fact]
		public void select_top_1_name()
		{
			var sql = "select top 1 name";
			Parse(sql);
			ThenExprShouldBe("SELECT TOP 1 name");
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
		public void select_number_from_table_aliasName()
		{
			var sql = @"select 1 from dbo.customer tb1";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer AS tb1");
		}

		[Fact]
		public void select_number_from_dbo_table_where()
		{
			var sql = @"select 1 from dbo.customer where name=customFunc()";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE name = customFunc()");
		}

		[Fact]
		public void select_number_from_dbo_table_where_and()
		{
			var sql = @"select 1 from dbo.customer where a=1 and b=2";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 FROM dbo.customer WHERE a = 1 AND b = 2");
		}

		[Fact]
		public void select_number_union_all_select_number()
		{
			var sql = @"select 1 
union all
select 2";
			Parse(sql);
			ThenExprShouldBe("SELECT 1 UNION ALL SELECT 2");
		}
	}
}
