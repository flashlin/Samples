using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ExistsParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);

			var arguments = new List<SqlCodeExpr>();
			var innerExpr = parser.ParseExp() as SqlCodeExpr;
			arguments.Add(innerExpr);

			var existsNameExpr = new ObjectIdSqlCodeExpr
			{
				DatabaseName = string.Empty,
				SchemaName = string.Empty,
				ObjectName = "EXISTS"
			};

			parser.Scanner.ConsumeAny(SqlToken.RParen);
			return new FuncSqlCodeExpr
			{
				Name = existsNameExpr,
				Parameters = arguments
			};
		}
	}

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