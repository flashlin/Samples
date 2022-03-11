using ExpectedObjects;
using PreviewLibrary;
using Xunit;

namespace TestProject
{
	public class TokenizerTest
	{
		[Fact]
		public void token()
		{
			var sql = "select DB_NAME";
			var token = new SqlTokenizer();
			token.PredicateParse(sql);
			token.Move();
			"DB_NAME".ToExpectedObject().ShouldEqual(token.Text);
		}
	}
}