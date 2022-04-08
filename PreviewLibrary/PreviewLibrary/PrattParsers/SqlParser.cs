using PreviewLibrary.Exceptions;
using PreviewLibrary.PrattParsers.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;


using Parselet = System.Func<
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

		public string Peek()
		{
			return _scanner.PeekString();
		}

		public TextSpan Consume(string expect = null)
		{
			return _scanner.Consume(expect);
		}

		public TextSpan Consume(SqlToken expectToken)
		{
			var token = _scanner.Consume();
			if( token.Type != expectToken)
			{
				var tokenStr = _scanner.GetSpanString(token);
				throw new ParseException($"Expect {expectToken}, but got '{tokenStr}'.");
			}
			return token;
		}

		public bool TryConsume(string expectToken, out TextSpan token)
		{
			if (!Match(expectToken))
			{
				token = TextSpan.Empty;
				return false;
			}
			token = _scanner.Consume();
			return true;
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

		public SqlDom ParseBy(Parselet parse)
		{
			var sqlDom = parse(this);
			if( sqlDom == null)
			{
				throw new ParseException();
			}
			return sqlDom;
		}
		
		public bool TryParseBy(Parselet parse, out SqlDom sqlDom)
		{
			sqlDom = parse(this);
			return sqlDom != null;
		}

		public SqlDom ParseBy(SqlToken expectPrefixToken)
		{
			var prefixToken = _scanner.Consume();
			if (prefixToken.IsEmpty)
			{
				throw new ParseException($"expect token but found NONE");
			}

			if (prefixToken.Type != expectPrefixToken)
			{
				var prefixTokenStr = _scanner.GetSpanString(prefixToken);
				throw new ParseException($"expect TokenType='{expectPrefixToken}' but found '{prefixTokenStr}' as TokenType='{prefixToken.Type}'");
			}

			var prefixParse = SqlSpec.Instance.Prefix(expectPrefixToken);
			return prefixParse(prefixToken, this);
		}

		public SqlDom ParseByAny(params SqlToken[] expectPrefixToken)
		{
			for (var i = 0; i < expectPrefixToken.Length; i++)
			{
				var prefixToken = expectPrefixToken[i];
				if (TryParseBy(prefixToken, out SqlDom sqlDom))
				{
					return sqlDom;
				}
			}
			throw new ParseException();
		}

		public bool TryParseBy<TSqlDom>(SqlToken expectPrefixToken, out TSqlDom sqlDom)
			where TSqlDom : SqlDom
		{
			var startIndex = _scanner.GetOffset();
			try
			{
				sqlDom = (TSqlDom)ParseBy(expectPrefixToken);
				return true;
			}
			catch (ParseException)
			{
				_scanner.SetOffset(startIndex);
				sqlDom = default;
				return false;
			}
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
				throw new ParseException($"Prefix '{tokenStr}' Parse fail.\r\n" + message);
			}

			return ProcessPrefix(prefixToken, prefixParse, ctxPrecedence);
		}

		public ParseException CreateParseException(TextSpan currentSpan)
		{
			return _scanner.CreateParseException(currentSpan);
		}

		private SqlDom ProcessPrefix(TextSpan prefixToken, PrefixParselet prefixParse, int ctxPrecedence)
		{
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
