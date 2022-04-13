using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
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