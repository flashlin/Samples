using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class PrintParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var valueExpr = parser.ParseExpIgnoreComment();
			return new PrintSqlCodeExpr
			{
				Value = valueExpr
			};
		}
	}
}
