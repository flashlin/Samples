using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class CteTest : SqlTestBase
	{
		public CteTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void with_tb1_as()
		{
			var sql = @"with temp as (select 1)";

			var expr = _sqlParser.ParseCtePartial(sql);

			@"WITH temp
AS (
SELECT 1
)".ShouldEqual(expr);
		}
	}
	}