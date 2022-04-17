using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ExecParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.Scanner.IsToken(SqlToken.Variable))
			{
				var valueExpr = ConsumeVariableAssignValueExpr(parser) as SqlCodeExpr;
				return new ExecSqlCodeExpr
				{
					ExecToken = "EXEC",
					Parameters = new[] { valueExpr }.ToList()
				};
			}

			var funcName = parser.ConsumeAny(SqlToken.SqlIdentifier, SqlToken.Identifier) as SqlCodeExpr;

			var parameters = new List<SqlCodeExpr>();
			do
			{
				var name = parser.ParseExpIgnoreComment();
				var isOutput = false;
				if (parser.Scanner.Match(SqlToken.Out))
				{
					isOutput = true;
				}
				parameters.Add(new ParameterSqlCodeExpr
				{
					Name = name,
					IsOutput = isOutput
				});
			} while (parser.Scanner.Match(SqlToken.Comma));


			return new ExecSqlCodeExpr
			{
				ExecToken = "EXEC",
				Name = funcName,
				Parameters = parameters
			};
		}

		private AssignSqlCodeExpr ConsumeVariableAssignValueExpr(IParser parser)
		{
			var variableSpan = parser.Scanner.Consume(SqlToken.Variable);
			var name = parser.PrefixParse(variableSpan, int.MaxValue) as SqlCodeExpr;
			parser.Scanner.Consume(SqlToken.Equal);
			var valueExpr = parser.ParseExpIgnoreComment();
			return new AssignSqlCodeExpr
			{
				Left = name,
				Right = valueExpr,
			};
		}
	}
}