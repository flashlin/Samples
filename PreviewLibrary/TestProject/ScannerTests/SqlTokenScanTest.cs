using Xunit;
using Xunit.Abstractions;
using FluentAssertions;

namespace TestProject.ScannerTests
{
	public class SqlTokenScanTest : ScanTestBase
	{
		public SqlTokenScanTest(ITestOutputHelper outputHelper) : base(outputHelper)
		{
		}

		[Fact]
		public void number()
		{
			var sql = "123";
			Scan(sql);

			var token = _scanner.Consume();
			token.ToString().Should().Be("123");
		}
	}
}
