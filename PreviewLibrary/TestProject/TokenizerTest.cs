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
		public void sql_float()
		{
			var token = GetToken("0.0010 xxx");
			"0.0010".ToExpectedObject().ShouldEqual(token);
		}

		private string GetToken(string text)
		{
			var token = new SqlTokenizer();
			token.PredicateParse(text);
			return token.Text;
		}
	}
}