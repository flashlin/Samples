using System;
using System.Collections.Generic;

namespace T1.CodeDom.Core
{
	public class PrattParser : IParser
	{
		protected readonly IScanner _scanner;
		private Dictionary<string, IPrefixParselet> _prefixParselets = new Dictionary<string, IPrefixParselet>();
		private Dictionary<string, IInfixParselet> _infixParselets = new Dictionary<string, IInfixParselet>();
		private List<InfixParseletInfo> _stashInfixParselets = new List<InfixParseletInfo>();	

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

		public virtual IExpression ParseExp(int ctxPrecedence)
		{
			var expr = GetParseExp(ctxPrecedence);
			if (expr == null)
			{
				throw new ParseException($"Expect token but found NONE.");
			}
			return expr;
		}

		public IExpression PrefixParse(TextSpan prefixToken, int ctxPrecedence)
		{
			var prefixParselet = CodeSpecPrefix(prefixToken);
			var left = prefixParselet.Parse(prefixToken, this);
			return PrefixParse(left, ctxPrecedence);
		}

		public IExpression PrefixParse(IExpression left, int ctxPrecedence)
		{
			while (true)
			{
				var (infixToken, consumeIndex) = PeekToken();
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
				ConsumeToken(consumeIndex);
				left = infixParselet.Parse(left, infixToken, this);
			}
			return left;
		}

		private void ConsumeToken(int consumeIndex)
		{
			_scanner.SetOffset(consumeIndex);
		}

		protected virtual (TextSpan infixToken, int consumeIndex) PeekToken()
		{
			var startIndex = _scanner.GetOffset();
			var infixToken = _scanner.Consume();
			var consumeIndex = _scanner.GetOffset();
			_scanner.SetOffset(startIndex);
			return (infixToken, consumeIndex);
		}

		public IEnumerable<IExpression> ParseProgram()
		{
			while (!_scanner.Peek().IsEmpty)
			{
				yield return ParseExp(0);
			}
		}

		public bool TryGetPrefixParselet(out IPrefixParselet parselet, TextSpan token)
		{
			return _prefixParselets.TryGetValue(token.Type, out parselet);
		}		
		
		public bool TryGetInfixParselet(out IInfixParselet parselet, TextSpan token)
		{
			return _infixParselets.TryGetValue(token.Type, out parselet);
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
		
		protected bool IsPrefixExists(TextSpan token)
		{
			return _prefixParselets.ContainsKey(token.Type);
		}

		protected void Register(string tokenType, IPrefixParselet parselet)
		{
			_prefixParselets.Add(tokenType, parselet);
		}

		protected void Register(string tokenType, IInfixParselet parselet)
		{
			_infixParselets.Add(tokenType, parselet);
		}

		public void StashInfixParselet<TTokenType>(TTokenType tokenType)
			where TTokenType : struct
		{
			var tokenTypeString = tokenType.ToString();
			var parselet = _infixParselets[tokenTypeString];
			_stashInfixParselets.Add(new InfixParseletInfo()
			{
				TokenType = tokenTypeString,
				Parselet = parselet
			});
			_infixParselets.Remove(tokenTypeString);
		}

		public void UnStashInfixParselet()
		{
			foreach (var info in _stashInfixParselets)
			{
				_infixParselets.Add(info.TokenType, info.Parselet);
			}
			_stashInfixParselets.Clear();
		}
	}

	public class InfixParseletInfo
	{
		public string TokenType { get; set; }		
		public IInfixParselet Parselet { get; set; }
	}
}
