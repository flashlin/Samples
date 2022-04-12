using Xunit.Abstractions;
using Xunit;

namespace TestProject.PrattTests
{
	public class FuncTest : TestBase
	{
		public FuncTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void exists()
		{
			var sql = @"exists(1)";
			Parse(sql);

			ThenExprShouldBe(@"EXISTS( 1 )");
		}

		[Fact]
		public void cast_hex()
		{
			var sql = @"cast(0x0FB AS DateTime)";
			Parse(sql);

			ThenExprShouldBe(@"CAST( 0x0FB AS DATETIME )");
		}

		[Fact]
		public void cast_float()
		{
			var sql = @"cast(100.00 as Decimal(19, 2))";
			Parse(sql);

			ThenExprShouldBe(@"CAST( 100.00 AS DECIMAL(19,2) )");
		}


	}
}
