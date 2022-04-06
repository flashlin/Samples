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

		[Fact]
		public void lparen_number()
		{
			var sql = "(1";
			Scan(sql);
			ThenTokenShouldBe("(");
		}

		[Fact]
		public void add_space()
		{
			var sql = "+ ";
			Scan(sql);
			ThenTokenShouldBe("+");
		}

		[Fact]
		public void multiComment()
		{
			var sql = " /* 12*34 **/";
			Scan(sql);
			ThenTokenShouldBe("/* 12*34 **/");
		}

		protected void ThenTokenShouldBe(string expected)
		{
			var token = _scanner.Consume();
			var tokenStr = _scanner.GetSpanString(token);
			tokenStr.Should().Be(expected);
		}
	}
}
