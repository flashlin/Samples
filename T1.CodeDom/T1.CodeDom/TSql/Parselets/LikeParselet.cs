using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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