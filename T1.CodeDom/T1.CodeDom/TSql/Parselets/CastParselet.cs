using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class CastParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);

			var expr = parser.ParseExp() as SqlCodeExpr;

			parser.Scanner.Consume(SqlToken.As);

			var datetype = parser.ConsumeDataType();

			parser.Scanner.Consume(SqlToken.RParen);

			var asExpr = new AsSqlCodeExpr
			{
				Left = expr,
				Right = datetype,
			};

			var parameters = new List<SqlCodeExpr>();
			parameters.Add(asExpr);

			return new FuncSqlCodeExpr
			{
				Name = new ObjectIdSqlCodeExpr
				{
					ObjectName = "CAST"
				},
				Parameters = parameters
			};
		}
	}
}