using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class ScanTest : TestBase
	{
		public ScanTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void script_setvar()
		{
			var sql = ":setvar";
			Scan(sql);
			ThenTokenShouldBe(":setvar");
		}

		[Fact]
		public void quote_string()
		{
			var sql = " \"12\\\"34\" ";
			Scan(sql);
			ThenTokenShouldBe("\"12\\\"34\"");
		}
	}
}
