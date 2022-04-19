using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.Net.SoapProtocols.WsdlXmlDeclrs;

namespace T1.CodeDom.TSql.Parselets
{
	public class NumberParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			return new NumberSqlCodeExpr
			{
				Value = tokenStr
			};
		}
	}
}
