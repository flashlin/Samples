using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.Core
{
	public static class IParserExtension
	{
		public static bool TryConsumeAny<TTokenType>(this IParser parser, out IExpression expr, params TTokenType[] tokenTypes)
			where TTokenType : struct
		{
			if (parser.Scanner.TryConsumeAny(out var token, tokenTypes))
			{
				expr = parser.PrefixParse(token, 0);
				return true;
			}
			expr = null;
			return false;
		}

		public static bool TryConsume<TTokenType>(this IParser parser, TTokenType tokenType, out IExpression expr)
			where TTokenType : struct
		{
			return TryConsumeAny<TTokenType>(parser, out expr, tokenType);
		}

		public static IExpression ConsumeAny<TTokenType>(this IParser parser, params TTokenType[] tokenTypes)
			where TTokenType : struct
		{
			if (!TryConsumeAny(parser, out var expr, tokenTypes))
			{
				var tokensStr = string.Join(",", tokenTypes.Select(x => x.ToString()));
				throw new ParseException($"Expect one of {tokensStr}");
			}
			return expr;
		}

		public static IExpression PrefixParse<TTokenType>(this IParser parser, TTokenType tokenType)
			where TTokenType : struct
		{
			var token = parser.Scanner.ConsumeAny(tokenType);
			return parser.PrefixParse(token, 0);
		}

		public static IExpression PrefixParseAny<TTokenType>(this IParser parser, params TTokenType[] tokenTypes)
			where TTokenType : struct
		{
			var token = parser.Scanner.ConsumeAny(tokenTypes);
			return parser.PrefixParse(token, 0);
		}

		public static IEnumerable<TExpression> ConsumeByDelimiter<TTokenType, TExpression>(this IParser parser, 
			TTokenType delimiter, 
			Func<TExpression> predicateExpr)
			where TTokenType : struct
			where TExpression : IExpression
		{
			do
			{
				var expr = predicateExpr();
				yield return expr;
			} while (parser.Scanner.TryConsume(delimiter, out _));
		}
	}
}
