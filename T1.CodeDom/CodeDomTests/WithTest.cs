using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
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
	}
}
