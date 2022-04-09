using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core.Parselets;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.Core
{
	public class PrattParser : IParser
	{
		protected readonly IScanner _scanner;
		private Dictionary<int, PrefixParselet> _prefixParselets = new Dictionary<int, PrefixParselet>();
		private Dictionary<int, InfixParselet> _infixParselets = new Dictionary<int, InfixParselet>();

		public PrattParser(IScanner scanner)
		{
			_scanner = scanner;
		}

		public IScanner Scanner
		{
			get { return _scanner; }
		}

		public IExpression ParseExp(int ctxPrecedence)
		{
			var prefixToken = _scanner.Consume();
			if (prefixToken.IsEmpty)
			{
				throw new ParseException($"Expect token but found NONE.");
			}

			var prefixParselet = CodeSpecPrefix(prefixToken);
			var left = prefixParselet.Parse(prefixToken, this);
			while (true)
			{
				var infixToken = _scanner.Peek();
				if (infixToken.IsEmpty)
				{
					break;
				}

				var infixParselet = CodeSpecInfix(infixToken.Type);
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

		protected virtual InfixParselet CodeSpecInfix(int tokenType)
		{
			return _infixParselets[tokenType];
		}

		protected virtual PrefixParselet CodeSpecPrefix(TextSpan token)
		{
			return _prefixParselets[token.Type];
		}

		protected void Register(int tokenType, PrefixParselet parselet)
		{
			_prefixParselets.Add(tokenType, parselet);
		}
	}
}
