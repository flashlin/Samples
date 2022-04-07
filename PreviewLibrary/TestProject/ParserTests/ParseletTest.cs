using FluentAssertions;
using PreviewLibrary.PrattParsers;
using PreviewLibrary.PrattParsers.Expressions;
using Xunit;

using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;
namespace TestProject.ParserTests
{
	public class ParseletTest
	{
		SqlDom _exp;

		[Fact]
		public void identifier()
		{
			var sql = "d123";

			Prefix(Parselets.Identifier, sql);

			ThenExprShouldBe("d123");
		}

		private void ThenExprShouldBe(string expected)
		{
			_exp.ToString().Should().Be(expected);
		}

		private void Prefix(PrefixParselet prefixParselet, string sql)
		{
			var scanner = new StringScanner(sql);
			var parser = new SqlParser(scanner);
			var head = parser.Consume();
			_exp = prefixParselet(head, parser);
		}
	}
}
