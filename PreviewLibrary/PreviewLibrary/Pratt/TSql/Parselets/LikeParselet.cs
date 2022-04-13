using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class LikeParselet : IInfixParselet
	{
		public IExpression Parse(IExpression left, TextSpan token, IParser parser)
		{
			var right = parser.ParseExp();
			return new LikeSqlCodeExpr
			{
				Left = left as SqlCodeExpr,
				Right = right as SqlCodeExpr
			};
		}

		public int GetPrecedence()
		{
			return (int)Precedence.COMPARE;
		}
	}
}