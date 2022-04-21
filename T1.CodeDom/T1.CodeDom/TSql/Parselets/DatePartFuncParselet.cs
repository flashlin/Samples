using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DatePartFuncParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.ConsumeToken(SqlToken.LParen);

			var datepartSpan = parser.ConsumeToken();
			var datepart = parser.Scanner.GetSpanString(datepartSpan);
			if (!DateAddFuncParselet.IsDatepartToken(datepart))
			{
				ThrowHelper.ThrowParseException(parser, $"Invalid datepart '{datepart}'.");
			}
			parser.ConsumeToken(SqlToken.Comma);

			var dateExpr = parser.ParseExpIgnoreComment();
			parser.ConsumeToken(SqlToken.RParen);

			return new FuncSqlCodeExpr
			{
				Name = new ObjectIdSqlCodeExpr
				{
					ObjectName = "DATEPART"
				},
				Parameters = new List<SqlCodeExpr>
				{
					new ConstantSqlCodeExpr
					{
						Value = datepart
					},
					dateExpr
				}
			};
		}
	}
}