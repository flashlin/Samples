using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class ReturnTest : SqlTestBase
	{
		public ReturnTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void return_none()
		{
			var sql = "return";
			
			var expr = _sqlParser.ParseReturnPartial(sql);
			
			"RETURN".ShouldEqual(expr);
		}
	}
}