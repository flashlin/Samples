using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class WhileParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var boolExpr = parser.ParseExpIgnoreComment();
			var body = parser.ConsumeBeginBody();
			return new WhileSqlCodeExpr
			{
				BooleanExpr = boolExpr,
				Body = body
			};
		}
	}
}