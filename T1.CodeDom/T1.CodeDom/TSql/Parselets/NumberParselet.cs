using System.Collections.Generic;
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

	public class ValuesParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var itemList = new List<SqlCodeExpr>();
			parser.ConsumeToken(SqlToken.LParen);
			do
			{
				var item = parser.ParseExpIgnoreComment();
				itemList.Add(item);
			} while (parser.MatchToken(SqlToken.Comma));
			parser.ConsumeToken(SqlToken.RParen);

			return new ValuesSqlCodeExpr
			{
				ValueList = itemList
			};
		}
	}
}
