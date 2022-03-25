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
	}
}