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
		public void select_from_table_left_join_from()
		{
			var sql = @"select 1 
from customer as c
left join secondTable,
otherTable
where c.id = 1";
			Parse(sql);
			ThenExprShouldBe(@"SELECT 1
FROM customer AS c
LEFT JOIN secondTable, 
otherTable
WHERE c.id = 1");
		}

		[Fact]
		public void left_join_nolock()
		{
			var sql = @"select 1 from customer
left join secondTable tb2 with(nolock)";

			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN secondTable tb2 WITH(NOLOCK)");
		}

		[Fact]
		public void left_join_where()
		{
			var sql = @"select 1 from customer
left join secondTable tb2 with(NOLOCK)
where id = 1";

			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer
LEFT JOIN secondTable tb2 WITH(NOLOCK)
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

		[Fact]
		public void nolock_join_table()
		{
			var sql = @"select 1
    	from customer c with (nolock)
     join otherTable o1";

			Parse(sql);

			ThenExprShouldBe(@"SELECT 1
FROM customer AS c WITH(NOLOCK)
JOIN otherTable o1");
		}

		[Fact]
		public void from_select_from_comment_inner_join()
		{
			var sql = @"select id
	from (
	 	select b.name
	 	from customer b with (nolock, index(pk_id)) -- test
		inner join otherTable e with (nolock) on b.id=e.id
	 	where birth < @birth
	 	group by b.id
	) a ";

			Parse(sql);

			ThenExprShouldBe(@"SELECT id
FROM ( SELECT b.name
FROM customer AS b WITH(NOLOCK, INDEX(PK_ID))
INNER JOIN otherTable e WITH(NOLOCK) b.id = e.id
WHERE birth < @birth
GROUP BY b.id ) AS a");
		}

	}
}