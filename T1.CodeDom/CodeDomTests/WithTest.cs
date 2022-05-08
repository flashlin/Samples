using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
{
	public class WithTest : TestBase
	{
		public WithTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void with_table_columns_as()
		{
			var sql = @"with tb1(id,name)
as (
	select id, name from customer
)";
			Parse(sql);

			ThenExprShouldBe(@"WITH tb1(id, name)
AS (
	SELECT id, name
	FROM customer
)");
		}

		[Fact]
		public void with_table_as()
		{
			var sql = @"with tmp
as (
	select id, name from customer
)";
			Parse(sql);

			ThenExprShouldBe(@"WITH tmp
AS (
	SELECT id, name
	FROM customer
)");
		}
		
		
		[Fact]
		public void with_over_order_by_sum()
		{
			var sql = @"with tmp
as (
	select ROW_NUMBER() OVER(ORDER BY Sum(Price) desc) AS ROWID, id from customer
)";
			Parse(sql);

			ThenExprShouldBe(@"WITH tmp
AS (
	SELECT ROW_NUMBER() OVER( ORDER BY SUM( Price ) DESC ) AS ROWID, id FROM customer
)");
		}
		
		
	}
}
