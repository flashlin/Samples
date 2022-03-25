using ExpectedObjects;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class FuncTest : SqlTestBase
	{
		public FuncTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void round()
		{
			var sql = "ROUND(748.58, -1, 1)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"ROUND( 748.58,-1,1 )".ToExpectedObject().ShouldEqual(expr.ToString());
		}

		[Fact]
		public void round_arithmetic()
		{
			var sql = "round(((@a - @b) * c), 0)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"round( @a - @b * c,0 )".ToExpectedObject().ShouldEqual(expr.ToString());
		}


		[Fact]
		public void sum()
		{
			var sql = "sum(a - b)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"sum( a - b )".ShouldEqual(expr);
		}
	}
}