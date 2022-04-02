using SqlDomTests.Helpers;
using System.Linq;
using T1.SqlDomParser;
using Xunit;
using Xunit.Abstractions;

namespace SqlDomTests
{
	public class ArithmeticTest : SqlTestBase
	{
		public ArithmeticTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void add()
		{
			var sql = "1 +2";
			var expr = _sqlParser.ParseSql(sql);
			expr.ShouldBe("1 + 2");
		}

		[Fact]
		public void mul()
		{
			var sql = "1 * 2";
			var expr = _sqlParser.ParseSql(sql);
			expr.ShouldBe("1 * 2");
		}

		[Fact]
		public void add_mul()
		{
			var sql = "1 + 2 * 3";
			var expr = _sqlParser.ParseSql(sql);
			expr.ShouldBe("1 + 2 * 3");
		}
	}
}