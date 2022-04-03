using Superpower;
using Superpower.Model;

namespace T1.SqlDomParser
{
	public class SqlTokenizer : Tokenizer<SqlToken>
	{
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
					var result = OneOf(
						(SqlToken.Identifier, SqlParser.Identifier)
					)(next);

					yield return result.parseResult;
					next = result.next;
				}
				else
				{
					var result = OneOf(
						SqlParser.CompareOps,
						SqlParser.BinaryOps
					)(next);

					yield return result.parseResult;
					next = result.next;
				}

				next = SkipWhiteSpace(next.Location);
			} while (next.HasValue);
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
	}
}
