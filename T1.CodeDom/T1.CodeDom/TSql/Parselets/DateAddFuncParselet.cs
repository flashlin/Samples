using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DateAddFuncParselet : IPrefixParselet
	{
		public static string[] datepartList = new[]
		{
			"year",
			"quarter",
			"month",
			"dayofyear",
			"day",
			"week",
			"weekday",
			"hour",
			"minute",
			"second",
			"millisecond",
			"microsecond",
			"nanosecond",
			"yy", "yyyy",
			"qq", "q",
			"mm", "m",
			"dy", "y",
			"dd", "d",
			"wk", "ww",
			"dw", "w",
			"hh",
			"mi", "n",
			"ss", "s",
			"ms",
			"mcs",
			"ns"
		};

		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.ConsumeToken(SqlToken.LParen);

			var datepartSpan = parser.ConsumeToken();
			var datepart = parser.Scanner.GetSpanString(datepartSpan);
			if (!IsDatepartToken(datepart))
			{
				ThrowHelper.ThrowParseException(parser, $"Invalid datepart '{datepart}'.");
			}
			parser.ConsumeToken(SqlToken.Comma);

			var numberExpr = parser.ParseExpIgnoreComment();
			parser.ConsumeToken(SqlToken.Comma);

			var dateExpr = parser.ParseExpIgnoreComment();
			parser.ConsumeToken(SqlToken.RParen);

			return new FuncSqlCodeExpr
			{
				Name = new ObjectIdSqlCodeExpr
				{
					ObjectName = "DATEADD"
				},
				Parameters = new List<SqlCodeExpr>
				{
					new ConstantSqlCodeExpr
					{
						Value = datepart
					},
					numberExpr,
					dateExpr
				}
			};
		}

		public static bool IsDatepartToken(string datepart)
		{
			if (datepartList.Select(x => x.ToUpper()).Contains(datepart.ToUpper()))
			{
				return true;
			}

			return (datepartList.Select(x => $"[{x.ToUpper()}]").Contains(datepart.ToUpper()));
		}
	}
}