using ExpectedObjects;
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
	}

	public class WhereTest : SqlTestBase
	{
		public WhereTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void where_a_and_b_eq_c()
		{
			var sql = "where a & b = c";
			var expr = _sqlParser.ParseWherePartial(sql);
			"a & b = c".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void where_a_is_null()
		{
			var sql = "where a is null";
			var expr = _sqlParser.ParseWherePartial(sql);
			"a IS NULL".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void a_like_b_and_or()
		{
			var sql = "WHERE desc LIKE 'a%' and b >= @c or b < @d";
			var expr = _sqlParser.ParseWherePartial(sql);
			"desc LIKE 'a%' and b >= @c or b < @d".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void a_in_select()
		{
			var sql = "WHERE id IN (SELECT pid FROM products)";
			var expr = _sqlParser.ParseWherePartial(sql);
			"id IN (SELECT pid FROM products)".ToExpectedObject().ShouldEqual(expr.ToString());
		}
	}
}