using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
			//var endExpr = ParseConstant(parser);
			var endExpr = parser.ParseExpIgnoreComment();
			endExpr = parser.ParseLRParenExpr(endExpr);

			return new BetweenSqlCodeExpr
			{
				LeftExpr = left as SqlCodeExpr,
				StartExpr = startExpr,
				EndExpr = endExpr,
			};
		}

		private static SqlCodeExpr ParseConstant(IParser parser)
		{
			return parser.PrefixParseAny(int.MaxValue, SqlToken.Number, SqlToken.QuoteString, SqlToken.HexNumber,
				SqlToken.Variable);
		}
	}
}