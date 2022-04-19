using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DoubleQuoteStringParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			var value = tokenStr.Substring(1, tokenStr.Length-2);
			return new DoubleQuoteStringSqlCodeExpr
			{
				Value = value
			};
		}
	}
}
