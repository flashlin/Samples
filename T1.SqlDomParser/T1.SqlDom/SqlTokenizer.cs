using Superpower;
using Superpower.Model;
using T1.SqlDom.Extensions;

namespace T1.SqlDomParser
{
	public class SqlTokenizer : Tokenizer<SqlToken>
	{
		static readonly SqlKeyword[] Keywords =
		{
			Word("and", SqlToken.And),
			Word("in", SqlToken.In),
			Word("is", SqlToken.Is),
			Word("like", SqlToken.Like),
			Word("not", SqlToken.Not),
			Word("or", SqlToken.Or),
			Word("true", SqlToken.True),
			Word("false", SqlToken.False),
			Word("null", SqlToken.Null)
		};

		static readonly Dictionary<string, SqlToken> KeywordsDict =
			Keywords.ToDictionary(x => x.Text.ToUpper(), x => x.Token);


		static readonly SqlKeyword[] Symbols =
		{
			Word(">=", SqlToken.GreaterThanOrEqual),
			Word("<=", SqlToken.LessThanOrEqual),
			Word("=", SqlToken.Equal),
			Word("<>", SqlToken.NotEqual),
			Word(">", SqlToken.GreaterThan),
			Word("<", SqlToken.LessThan),
			Word(".", SqlToken.Dot),
			Word("+", SqlToken.Plus),
			Word("-", SqlToken.Minus),
			Word("*", SqlToken.Star),
			Word("/", SqlToken.Divide),
		};

		static readonly Dictionary<string, SqlToken> SymbolsDict =
			Symbols.ToDictionary(x => x.Text.ToUpper(), x => x.Token);

		protected override IEnumerable<Result<SqlToken>> Tokenize(
				TextSpan stringSpan,
				TokenizationState<SqlToken> tokenizationState)
		{
			var next = SkipWhiteSpace(stringSpan);
			if (!next.HasValue)
				yield break;

			do
			{
				if (char.IsDigit(next.Value))
				{
					var result = OneOf(
						(SqlToken.HexNumber, SqlParser.HexInteger),
						(SqlToken.Number, SqlParser.Real)
					)(next);

					yield return result.parseResult;
					next = result.next;
				}
				else if (char.IsLetter(next.Value) || next.Value == '_')
				{
					var beginIdentifier = next.Location;
					do
					{
						next = next.Remainder.ConsumeChar();
					}
					while (next.HasValue && (char.IsLetterOrDigit(next.Value) || next.Value == '_'));

					if (TryGetKeyword(beginIdentifier.Until(next.Location), out var keyword))
					{
						yield return Result.Value(keyword, beginIdentifier, next.Location);
					}
					else
					{
						yield return Result.Value(SqlToken.Identifier, beginIdentifier, next.Location);
					}
				}
				else
				{
					var beginSymbol = next.Location;
					do
					{
						next = next.Remainder.ConsumeChar();
					}
					while (next.HasValue && !IsIdent(next));

					var symbolText = beginSymbol.Until(next.Location).ToStringValue().TrimSpaces();
					if (TryGetSymbol(symbolText, out var symbol))
					{
						yield return Result.Value(symbol, beginSymbol, next.Location);
					}
					else
					{
						yield return Result.Value(SqlToken.Unknown, beginSymbol, next.Location);
					}
				}

				next = SkipWhiteSpace(next.Location);
			} while (next.HasValue);
		}

		static bool IsIdent(Result<char> next)
		{
			return char.IsLetterOrDigit(next.Value) || next.Value == '_';
		}

		static bool TryGetKeyword(TextSpan span, out SqlToken keyword)
		{
			var spanText = span.ToStringValue().ToUpper();
			if (KeywordsDict.TryGetValue(spanText, out var sqlToken))
			{
				keyword = sqlToken;
				return true;
			}
			keyword = SqlToken.None;
			return false;
		}

		static bool TryGetSymbol(string spanText, out SqlToken keyword)
		{
			if (SymbolsDict.TryGetValue(spanText, out var sqlToken))
			{
				keyword = sqlToken;
				return true;
			}
			keyword = SqlToken.None;
			return false;
		}

		private Func<Result<char>, SqlToken> To(SqlToken sqlType, TextParser<string> parser)
		{
			return (Result<char> next) =>
			{
				var token = parser(next.Location);
				if (token.HasValue)
				{
					next = token.Remainder.ConsumeChar();
					var result = Result.Value(sqlType, token.Location, token.Remainder);
					//return (next, result);
					return result.Value;
				}
				var emptyResult = Result.Empty<SqlToken>(next.Location);
				next = next.Remainder.ConsumeChar();
				//return (next, emptyResult);
				return emptyResult.Value;
			};
		}

		private Func<Result<char>,
			(Result<char> next, Result<SqlToken> parseResult)> OneOf(params
			(SqlToken sqlType, dynamic parser)[] parsers)
		{
			return (Result<char> next) =>
			{
				for (var i = 0; i < parsers.Length; i++)
				{
					var item = parsers[i];
					var token = item.parser(next.Location);
					if (token.HasValue)
					{
						next = token.Remainder.ConsumeChar();
						var result = Result.Value(item.sqlType, token.Location, token.Remainder);
						return (next, result);
					}
				}

				var emptyResult = Result.Empty<SqlToken>(next.Location);
				next = next.Remainder.ConsumeChar();
				return (next, emptyResult);
			};
		}

		private Func<Result<char>,
			(Result<char> next, Result<SqlToken> parseResult)> OneOf(params
			TextParser<SqlToken>[] parsers)
		{
			return (Result<char> next) =>
			{
				for (var i = 0; i < parsers.Length; i++)
				{
					var parser = parsers[i];
					var token = parser(next.Location);
					if (token.HasValue)
					{
						next = token.Remainder.ConsumeChar();
						var result = Result.Value(token.Value, token.Location, token.Remainder);
						return (next, result);
					}
				}

				var emptyResult = Result.Empty<SqlToken>(next.Location);
				next = next.Remainder.ConsumeChar();
				return (next, emptyResult);
			};
		}

		private static SqlKeyword Word(string text, SqlToken tokenType)
		{
			return new SqlKeyword(text, tokenType);
		}
	}
}
