using FluentAssertions;
using System.Linq;
using T1.SqlDomParser;
using Xunit;

namespace SqlDomTests
{
	public class SqlTestBase
	{

	}

	public class ArithmeticTest
	{
		[Fact]
		public void add()
		{
			var sql = "1 +2";

			var expr = new SqlParser().ParseSql(sql);

			var arithmeticExpr = expr as BinaryExpr;

			arithmeticExpr!.ShouldBe("1 + 2");
		}

		[Fact]
		public void mul()
		{
			var sql = "1 * 2";

			var expr = new SqlParser().ParseSql(sql);

			var arithmeticExpr = expr as BinaryExpr;

			arithmeticExpr!.ShouldBe("1 * 2");
		}

		[Fact]
		public void add_mul()
		{
			var sql = "1 + 2 * 3";

			var expr = new SqlParser().ParseSql(sql);

			var arithmeticExpr = expr as BinaryExpr;

			arithmeticExpr!.ShouldBe("1 + 2 * 3");
		}
	}

	public static class TestExtensions
	{
		public static void ShouldBe(this SqlExpr sqlExpr, string expected)
		{
			sqlExpr.Should().NotBeNull();
			sqlExpr.ToSqlCode().Should().Be(expected);
		}
	}
}