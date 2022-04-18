using SqlDomTests.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace SqlDomTests
{
	public class CompareTest : SqlTestBase
	{
		public CompareTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void greaterThan_and_or()
		{
			var sql = "a > 1 and b >=2 or c = 3";
			var expr = _sqlParser.ParseSql(sql);
			expr.ShouldBe("");
		}

		[Fact]
		public void eq_and_or()
		{
			var sql = "a+1=1 and b*2=3 or c/3=4";
			var expr = _sqlParser.ParseSql(sql);
			expr.ShouldBe("");
		}

		[Fact]
		public void identifier_add_number_eq_number()
		{
			var sql = "a+1=1";
			var expr = _sqlParser.ParseSql(sql);
			expr.ShouldBe("");
		}


	}
}