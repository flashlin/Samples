using PreviewLibrary.PrattParsers.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;


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

		public TextSpan Consume(string expect = null)
		{
			return _scanner.Consume(expect);
		}

		public bool TryConsume(SqlToken expectToken, out TextSpan token)
		{
			if (!Match(expectToken))
			{
				token = TextSpan.Empty;
				return false;
			}
			token = _scanner.Consume();
			return true;
		}

		public bool TryConsumes(out List<TextSpan> tokenList, params SqlToken[] expectTokens)
		{
			var startIndex = _scanner.GetOffset();
			tokenList = new List<TextSpan>();
			foreach (var expectToken in expectTokens)
			{
				if (!Match(expectToken))
				{
					_scanner.SetOffset(startIndex);
					tokenList = new List<TextSpan>();
					return false;
				}
				var token = _scanner.Consume();
				tokenList.Add(token);
			}
			return true;
		}

		public bool TryConsumes(out List<TextSpan> tokenList, params SqlToken[][] expectTokens)
		{
			var startIndex = _scanner.GetOffset();
			tokenList = new List<TextSpan>();
			foreach (var anyExpectTokens in expectTokens)
			{
				if (!anyExpectTokens.Any(expect => Match(expect)))
				{
					_scanner.SetOffset(startIndex);
					tokenList = new List<TextSpan>();
					return false;
				}
				var token = _scanner.Consume();
				tokenList.Add(token);
			}
			return true;
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

			PrefixParselet prefixParse = null;

			try
			{
				prefixParse = SqlSpec.Instance.Prefix(prefixToken.Type);
			}
			catch (KeyNotFoundException)
			{
				var tokenStr = _scanner.GetSpanString(prefixToken);
				var message = _scanner.GetHelpMessage(prefixToken);
				throw new Exception($"Prefix '{tokenStr}' Parse fail.\r\n" + message);
			}

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
