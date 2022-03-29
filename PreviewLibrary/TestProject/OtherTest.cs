using ExpectedObjects;
using TestProject.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace TestProject
{
	public class OtherTest : SqlTestBase
	{
		public OtherTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void waitfor_delay()
		{
			var sql = "waitfor delay '00:00:00.300'";
			var expr = _sqlParser.Parse(sql);
			"WAITFOR DELAY '00:00:00.300'".ShouldEqual(expr);
		}
	}

}