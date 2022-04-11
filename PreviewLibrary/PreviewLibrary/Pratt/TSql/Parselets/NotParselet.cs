using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class NotParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
         var right = parser.ParseExp() as SqlCodeExpr;
         return new NotSqlCodeExpr
			{
            Right = right,
			};
		}
	}
}