using PreviewLibrary.Exceptions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.Core
{
	public class PrattParser<TExpr> : IParser
	{
		protected readonly IScanner _scanner;
		private Dictionary<int, PrefixParselet<TExpr>> _prefixParselets = new Dictionary<int, PrefixParselet<TExpr>>();
		private Dictionary<int, InfixParselet<TExpr>> _infixParselets = new Dictionary<int, InfixParselet<TExpr>>();

		public PrattParser(IScanner scanner)
		{
			_scanner = scanner;
		}

		public TExpr ParseExp(int ctxPrecedence)
		{
			var prefixToken = _scanner.Consume();
			if (prefixToken.IsEmpty)
			{
				throw new ParseException($"expect token but found NONE");
			}

			var prefixParselet = CodeSpecPrefix(prefixToken.Type);
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

		public IEnumerable<TExpr> ParseProgram()
		{
			while (!_scanner.Peek().IsEmpty)
			{
				yield return ParseExp(0);
			}
		}

		protected InfixParselet<TExpr> CodeSpecInfix(int tokenType)
		{
			return _infixParselets[tokenType];
		}

		protected PrefixParselet<TExpr> CodeSpecPrefix(int tokenType)
		{
			return _prefixParselets[tokenType];
		}
	}
}
