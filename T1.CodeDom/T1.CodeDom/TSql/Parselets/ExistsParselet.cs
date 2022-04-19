using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
}