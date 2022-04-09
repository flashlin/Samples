using PreviewLibrary.Pratt.Core.Expressions;

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

		public static IExpression PrefixParse<TTokenType>(this IParser parser, TTokenType tokenType)
			where TTokenType : struct
		{
			var token = parser.Scanner.ConsumeAny(tokenType);
			return parser.PrefixParse(token, 0);
		}
	}
}
