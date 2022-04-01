using ExpectedObjects;
using TestProject.Helpers;
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
		public void unknown_custom_func()
		{
			var sql = "strsplitmax(@a, N',')";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"strsplitmax( @a,N',' )".ShouldEqual(expr);
		}

		[Fact]
		public void cast()
		{
			var sql = "cast(@a as date)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"CAST( @a AS date )".ShouldEqual(expr);
		}

		[Fact]
		public void cast_otherFunc_arithmetic_as_date()
		{
			var sql = "cast( getdate()+1 as date)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"CAST( getdate() + 1 AS date )".ShouldEqual(expr);
		}




		[Fact]
		public void isnull()
		{
			var sql = "isnull(@betCondition, '')";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"isnull( @betCondition,'' )".ShouldEqual(expr);
		}

		[Fact]
		public void exists_select()
		{
			var sql = @"exists(
								SELECT 1 FROM @a
								WHERE name = 
									CAST( @b AS nvarchar(3) ) + ':' + CAST( @c AS nvarchar(3) ) 
							)";

			var expr = _sqlParser.ParseFuncPartial(sql);

			"exists( SELECT 1 FROM @a WHERE name = CAST( @b AS nvarchar(3) ) + ':' + CAST( @c AS nvarchar(3) ) )"
				.ShouldEqual(expr);
		}

		[Fact]
		public void round()
		{
			var sql = "ROUND(748.58, -1, 1)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"ROUND( 748.58,-1,1 )".ShouldEqual(expr);
		}

		[Fact]
		public void round_arithmetic()
		{
			var sql = "round(((@a - @b) * c), 0)";
			var expr = _sqlParser.ParseFuncPartial(sql);

			"round( ((@a - @b) * c),0 )".ShouldEqual(expr);
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