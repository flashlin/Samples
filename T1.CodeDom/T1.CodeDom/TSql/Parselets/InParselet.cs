using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class InParselet : IInfixParselet
	{
		public int GetPrecedence()
		{
			return (int)Precedence.COMPARE;
		}

		public IExpression Parse(IExpression left, TextSpan token, IParser parser)
		{
			var rightExpr = parser.ConsumeValueList();
			return new InSqlCodeExpr
			{
				Left = left as SqlCodeExpr,
				Right = rightExpr
			};
		}

		private static SqlCodeExpr ParseConstantList(SqlCodeExpr constantExpr0, IParser parser)
		{
			var constantList = new List<SqlCodeExpr>();
			constantList.Add(constantExpr0);
			while (parser.Scanner.Match(SqlToken.Comma))
			{
				if (!parser.TryConsumeAny(out var constantExpr, SqlToken.Number, SqlToken.NString, SqlToken.QuoteString))
				{
					break;
				}
				constantList.Add(constantExpr as SqlCodeExpr);
			}
			return new ExprListSqlCodeExpr
			{
				Items = constantList,
			};
		}
	}
}