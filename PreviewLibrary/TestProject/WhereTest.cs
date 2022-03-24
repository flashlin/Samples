using ExpectedObjects;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
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