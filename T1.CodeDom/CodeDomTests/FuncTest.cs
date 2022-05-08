using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
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

			ThenExprShouldBe(@"CAST( 100.00 AS DECIMAL (19,2) )");
		}

		[Fact]
		public void convert()
		{
			var sql = @"CONVERT(varchar(100), GETDATE(), 120)";

			Parse(sql);

			ThenExprShouldBe(@"CONVERT( VARCHAR (100), GETDATE(), 120 )");
		}

		[Fact]
		public void isnull()
		{
			var sql = @"isnull(@id, '')";

			Parse(sql);

			ThenExprShouldBe(@"ISNULL( @id, '' )");
		}

		[Fact]
		public void getdate()
		{
			var sql = @"getdate()";

			Parse(sql);

			ThenExprShouldBe(@"GETDATE()");
		}

		[Fact]
		public void floor()
		{
			var sql = @"floor(1)";

			Parse(sql);

			ThenExprShouldBe(@"FLOOR( 1 )");
		}
		
		
		[Fact]
		public void char_fn()
		{
			var sql = @"char(@code)";

			Parse(sql);

			ThenExprShouldBe(@"CHAR( @code )");
		}
		
		
	}
}
