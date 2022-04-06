using PreviewLibrary.PrattParsers.Expressions;
using System;
using System.Collections.Generic;

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
			return _scanner.Match(expect);
		}

		public bool Match(SqlToken expectToken)
		{
			return _scanner.Match(expectToken);
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
				throw new Exception($"expect token but found NONE");
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

		public IEnumerable<SqlDom> ParseProgram()
		{
			while (!_scanner.Peek().IsEmpty)
			{
				yield return ParseExp(0);
			}
			//return ParseExp(0);
		}
	}
}
