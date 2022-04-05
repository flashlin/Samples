using PreviewLibrary.PrattParsers.Expressions;
using System;

namespace PreviewLibrary.PrattParsers
{
	public class SqlParser : IParser
	{
		private readonly IScanner _scanner;

		public SqlParser(IScanner scanner)
		{
			this._scanner = scanner;
		}

		public bool Match(string expect)
		{
			return _scanner.MatchIgnoreCase(expect);
		}

		public void Consume(string expect)
		{
			_scanner.Consume(expect);
		}

		public string GetSpanString(TextSpan span)
		{
			return _scanner.GetSpanString(span);
		}

		public SqlDom ParseExp(int ctxPrecedence)
		{
			var prefixToken = _scanner.Consume();
			if (prefixToken.IsEmpty)
			{
				throw new Exception($"expect token but found none");
			}

			var prefixParse = SqlSpec.Instance.Prefix(prefixToken.Type);

			var left = prefixParse(prefixToken, this);

			while (true)
			{
				var infixToken = _scanner.Peek();
				if (infixToken.IsEmpty)
				{
					break;
				}

				var infixParselet = SqlSpec.Instance.Infix(infixToken.Type);
				if (infixParselet.parse == null)
				{
					break;
				}

				if (infixParselet.precedence <= ctxPrecedence)
				{
					break;
				}
				_scanner.Consume();
				left = infixParselet.parse(infixToken, left, this);
			}
			return left;
		}

		public SqlDom ParseProgram()
		{
			return ParseExp(0);
		}
	}
}
