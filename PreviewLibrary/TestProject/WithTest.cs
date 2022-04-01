using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class WithTest : SqlTestBase
	{
		public WithTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void with_x_as_comma_x_as()
		{
			var sql = @"with a as ( select 1 ), b as (select 2)";

			var expr = _sqlParser.ParseWithAliasnameAsPartial(sql);

			@"WITH a AS ( SELECT 1 ),b AS ( SELECT 2 )".ShouldEqual(expr);
		}
	}
}