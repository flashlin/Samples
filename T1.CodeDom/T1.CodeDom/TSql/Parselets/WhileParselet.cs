using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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