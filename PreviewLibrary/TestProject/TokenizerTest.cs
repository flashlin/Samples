using ExpectedObjects;
using PreviewLibrary;
using Xunit;

namespace TestProject
{
	public class TokenizerTest
	{
		[Fact]
		public void sql_ident()
		{
			var token = GetToken("DB_NAME xxx");
			"DB_NAME".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void sql_int()
		{
			var token = GetToken("1 xxx");
			"1".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void sql_negative_int()
		{
			var token = GetToken("-1 xxx");
			"-".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void insert()
		{
			var token = GetToken("INSERT xxx");
			"INSERT".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void single_comment()
		{
			var token = GetToken("-- 123\r\nxxx");
			"xxx".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void sql_float()
		{
			var token = GetToken("0.0010 xxx");
			"0.0010".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void sigle_string()
		{
			var token = GetToken("N'aaa D''bbb'");
			"N'aaa D''bbb'".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void great_equal()
		{
			var token = GetToken("> =");
			">=".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void c_comment()
		{
			var token = GetToken("/* 123 */");
			"".ToExpectedObject().ShouldEqual(token);
		}

		[Fact]
		public void temp_table()
		{
			var token = GetToken("#tmp1");
			"#tmp1".ToExpectedObject().ShouldEqual(token);
		}

		private string GetToken(string text)
		{
			var token = new SqlTokenizer();
			token.PredicateParse(text);
			return token.Text;
		}
	}
}