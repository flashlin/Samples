using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.Core
{
	public class PrattParser : IParser
	{
		protected readonly IScanner _scanner;
		private Dictionary<string, IPrefixParselet> _prefixParselets = new Dictionary<string, IPrefixParselet>();
		private Dictionary<string, IInfixParselet> _infixParselets = new Dictionary<string, IInfixParselet>();

		public PrattParser(IScanner scanner)
		{
			_scanner = scanner;
		}

		public IScanner Scanner
		{
			get { return _scanner; }
		}

		public IExpression GetParseExp(int ctxPrecedence)
		{
			var prefixToken = _scanner.Consume();
			if (prefixToken.IsEmpty)
			{
				return null;
			}
			return PrefixParse(prefixToken, ctxPrecedence);
		}

		public IExpression ParseExp(int ctxPrecedence)
		{
			var expr = GetParseExp(ctxPrecedence);
			if (expr == null)
			{
				throw new ParseException($"Expect token but found NONE.");
			}
			return expr;
			//var prefixToken = _scanner.Consume();
			//if (prefixToken.IsEmpty)
			//{
			//	throw new ParseException($"Expect token but found NONE.");
			//}
			//return PrefixParse(prefixToken, ctxPrecedence);
		}

		public IExpression PrefixParse(TextSpan prefixToken, int ctxPrecedence)
		{
			var prefixParselet = CodeSpecPrefix(prefixToken);
			var left = prefixParselet.Parse(prefixToken, this);
			while (true)
			{
				var infixToken = _scanner.Peek();
				if (infixToken.IsEmpty)
				{
					break;
				}

				var infixParselet = CodeSpecInfix(infixToken);
				if (infixParselet == null)
				{
					break;
				}

				if (infixParselet.GetPrecedence() <= ctxPrecedence)
				{
					break;
				}
				_scanner.Consume();
				left = infixParselet.Parse(left, infixToken, this);
			}
			return left;
		}

		public IEnumerable<IExpression> ParseProgram()
		{
			while (!_scanner.Peek().IsEmpty)
			{
				yield return ParseExp(0);
			}
		}

		protected virtual IInfixParselet CodeSpecInfix(TextSpan token)
		{
			if (_infixParselets.TryGetValue(token.Type, out var infixParselet))
			{
				return infixParselet;
			}
			return null;
		}

		protected virtual IPrefixParselet CodeSpecPrefix(TextSpan token)
		{
			return _prefixParselets[token.Type];
		}

		protected void Register(string tokenType, IPrefixParselet parselet)
		{
			_prefixParselets.Add(tokenType, parselet);
		}

		protected void Register(string tokenType, IInfixParselet parselet)
		{
			_infixParselets.Add(tokenType, parselet);
		}
	}
}
