using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class BetweenParselet : IInfixParselet
	{
		public int GetPrecedence()
		{
			return (int)Precedence.COMPARE;
		}

		public IExpression Parse(IExpression left, TextSpan token, IParser parser)
		{
			var startExpr = ParseConstant(parser);
			parser.Scanner.Consume(SqlToken.And);
			var endExpr = ParseConstant(parser);

			return new BetweenSqlCodeExpr
			{
				LeftExpr = left as SqlCodeExpr,
				StartExpr = startExpr,
				EndExpr = endExpr,
			};
		}

		private static SqlCodeExpr ParseConstant(IParser parser)
		{
			return parser.PrefixParseAny(int.MaxValue, SqlToken.Number, SqlToken.QuoteString, SqlToken.HexNumber);
		}
	}
}