using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class QuoteStringParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			var value = tokenStr.Substring(1, tokenStr.Length - 2);
			return new QuoteStringSqlCodeExpr
			{
				Value = value
			};
		}
	}
}
