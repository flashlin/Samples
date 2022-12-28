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
			var sql = "select name from customer where a > 1 and b >=2 or c = 3";
			Parse(sql);
			ThenResultShouldBe("SELECT name FROM customer WHERE a > 1 AND b >=2 OR c = 3");
		}
	}
}