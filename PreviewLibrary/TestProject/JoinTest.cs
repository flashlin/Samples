using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class JoinTest : SqlTestBase
	{
		public JoinTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void left_outer_join()
		{
			var sql = @"select id from tb1
left outer join tb2 on tb2.id = tb1.id and tb2.p = tb1.p
where tb1.id = 1";

			var expr = _sqlParser.ParseSelectPartial(sql);

			"SELECT id FROM tb1 Left outer JOIN tb2 ON tb2.id = tb1.id and tb2.p = tb1.p WHERE tb1.id = 1"
				.ToExpectedObject().ShouldEqual(expr.ToString());
		}


		[Fact]
		public void many_join()
		{
			var sql = @"select id from tb1
left outer join tb2 on tb2.id = tb1.id
inner join tb3 on tb3.id = tb1.id
where tb1.id = 1";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT id FROM tb1 Left outer JOIN tb2 ON tb2.id = tb1.id
Inner JOIN tb3 ON tb3.id = tb1.id WHERE tb1.id = 1".ShouldEqual(expr);
		}

		[Fact]
		public void right_outer_join()
		{
			var sql = @"select field1 from customer as f
				right OUTER JOIN tb2 ON f.id = tb2.id";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT field1 FROM customer AS f Right OUTER JOIN tb2 ON f.id = tb2.id".ShouldEqual(expr);
		}

		[Fact]
		public void right_outer_join_select()
		{
			var sql = @"select field1 from customer as f
				right OUTER JOIN (
					SELECT field2, RANK() OVER(ORDER BY t.id DESC, t.price DESC) AS newRanking
					FROM @tb1, customer c with(nolock) where c.id = t.id
			   ) AS tb2 ON f.custid = newRanking.custid";

			var expr = _sqlParser.ParseSelectPartial(sql);

			@"SELECT field1 FROM customer AS f Right OUTER JOIN (SELECT field2,RANK() OVER(
ORDER BY t.id DESC,t.price DESC
) AS newRanking FROM @tb1,customer AS c WITH(nolock) WHERE c.id = t.id) AS tb2 ON f.custid = newRanking.custid"
			.ShouldEqual(expr);
		}


	}
}