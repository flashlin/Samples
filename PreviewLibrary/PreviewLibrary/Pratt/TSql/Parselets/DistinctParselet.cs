using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class DistinctParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var value = parser.ParseExpIgnoreComment();
			return new DistinctSqlCodeExpr()
			{
				Value = value,
			};
		}
	}
}
