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
			ThenTokenShouldBe("123");
		}

		[Fact]
		public void space_identifier()
		{
			var sql = " name";
			Scan(sql);
			ThenTokenShouldBe("name");
		}

		[Fact]
		public void space_symbol2()
		{
			var sql = " >=";
			Scan(sql);
			ThenTokenShouldBe(">=");
		}

		protected void ThenTokenShouldBe(string expected)
		{
			var token = _scanner.Consume();
			token.ToString().Should().Be(expected);
		}
	}
}
