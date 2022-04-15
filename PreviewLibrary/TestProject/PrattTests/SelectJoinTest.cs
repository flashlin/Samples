using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class SelectJoinTest : TestBase
	{
		public SelectJoinTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void left_join()
		{
			var sql = "select 1 from customer left join secondTable";
			Parse(sql);
			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN secondTable");
		}

		[Fact]
		public void left_join_nolock()
		{
			var sql = @"select 1 from customer
left join secondTable tb2 with(nolock)";

			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN secondTable tb2 WITH(nolock)");
		}

		[Fact]
		public void left_join_where()
		{
			var sql = @"select 1 from customer
left join secondTable tb2 with(nolock)
where id = 1";

			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN secondTable tb2 WITH(nolock)
WHERE id = 1");
		}

		[Fact]
		public void left_join_variableName()
		{
			var sql = @"select 1 from customer
left join @tb1
where id = 1";

			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN @tb1
WHERE id = 1");
		}


	}
}
