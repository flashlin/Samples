﻿using TestProject.PrattTests;
using Xunit;
using Xunit.Abstractions;

namespace CodeDomTests
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
		public void singleComment()
		{
			var sql = @"-- 123";
			Scan(sql);
			ThenTokenShouldBe("-- 123");
		}

		[Fact]
		public void nstring()
		{
			var sql = "N'123'";
			Scan(sql);
			ThenTokenShouldBe("N'123'");
		}

		[Fact]
		public void nstring_2_quote()
		{
			var sql = "N';'''";
			Scan(sql);
			ThenTokenShouldBe("N';'''");
		}

		[Fact]
		public void nstring_2_quote2()
		{
			var sql = "N';''', N'";
			Scan(sql);
			ThenTokenShouldBe("N';'''");
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

		[Fact]
		public void smaller_bigger_than()
		{
			var sql = "<>";
			Scan(sql);
			ThenTokenShouldBe("<>");
		}

		[Fact]
		public void colon_colon()
		{
			var sql = "::";
			Scan(sql);
			ThenTokenShouldBe("::");
		}

		[Fact]
		public void scan_all()
		{
			var sql = "OBJECT::";
			ScanAll(sql);
			ThenTokenListShouldBe("OBJECT", "::");
		}

		[Fact]
		public void biggerThan_equal()
		{
			var sql = ">  =1";
			ScanAll(sql);
			ThenTokenListShouldBe(">  =", "1");
		}

		[Fact]
		public void biggerThan_1()
		{
			var sql = ">  1";
			ScanAll(sql);
			ThenTokenListShouldBe(">", "1");
		}

		[Fact]
		public void crlf()
		{
			var sql = "1\r\n 2\n 3--a\n4";
			ScanAll(sql);
			ThenTokenListShouldBe("1", "2", "3", "--a", "4");
		}


		[Fact]
		public void temp()
		{
			var sql = "#tmp";
			ScanAll(sql);
			ThenTokenListShouldBe("#tmp");
		}
		
		
		[Fact]
		public void variable_xml()
		{
			var sql = "@xml";
			ScanAll(sql);
			ThenTokenListShouldBe("@xml");
		}
		
		
		[Fact]
		public void batch_variable()
		{
			var sql = "$(abc)";
			ScanAll(sql);
			ThenTokenListShouldBe("$(abc)");
		}
		
		
	}
}
