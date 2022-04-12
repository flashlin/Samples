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
		public void doubleQuoteString()
		{
			var sql = " \"12\\\"34\" ";
			Scan(sql);
			ThenTokenShouldBe("\"12\\\"");
		}

		[Fact]
		public void doubleQuoteString_contain_slash()
		{
			var sql = @" ""C:\abc"" ";
			Scan(sql);
			ThenTokenShouldBe("\"C:\\abc\"");
		}

		[Fact]
		public void quoteString_contain_quoteString()
		{
			var sql = @" '1''6' ";
			Scan(sql);
			ThenTokenShouldBe("'1''6'");
		}

		[Fact]
		public void multiComment()
		{
			var sql = @"/* 123 */";
			Scan(sql);
			ThenTokenShouldBe("/* 123 */");
		}

		[Fact]
		public void nstring()
		{
			var sql = "N'123'";
			Scan(sql);
			ThenTokenShouldBe("N'123'");
		}

		[Fact]
		public void lparent()
		{
			var sql = "(-";
			Scan(sql);
			ThenTokenShouldBe("(");
		}
		
		[Fact]
		public void var()
		{
			var sql = "@name";
			Scan(sql);
			ThenTokenShouldBe("@name");
		}

		[Fact]
		public void equal()
		{
			var sql = "=";
			Scan(sql);
			ThenTokenShouldBe("=");
		}

		[Fact]
		public void hex()
		{
			var sql = "0x013FA";
			Scan(sql);
			ThenTokenShouldBe("0x013FA");
		}

		[Fact]
		public void sql_float()
		{
			var sql = "0.0010";
			Scan(sql);
			ThenTokenShouldBe("0.0010");
		}

		[Fact]
		public void sql_float2()
		{
			var sql = "10.0010";
			Scan(sql);
			ThenTokenShouldBe("10.0010");
		}
	}
}
