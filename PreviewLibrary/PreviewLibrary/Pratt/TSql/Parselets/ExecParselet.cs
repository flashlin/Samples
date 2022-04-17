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
			SqlCodeExpr returnVariable = null;
			if (parser.Scanner.TryConsume(SqlToken.Variable, out var returnVariableSpan))
			{
				returnVariable = parser.PrefixParse(returnVariableSpan, int.MaxValue) as SqlCodeExpr;
				parser.Scanner.Consume(SqlToken.Equal);
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
				ReturnVariable = returnVariable,
				Name = funcName,
				Parameters = parameters
			};
		}

		//private List<AssignSqlCodeExpr> ConsumeVariableAssignValueExpr(IParser parser)
		//{
		//	var items = new List<AssignSqlCodeExpr>();
		//	do
		//	{
		//		var variableSpan = parser.Scanner.Consume(SqlToken.Variable);
		//		var name = parser.PrefixParse(variableSpan, int.MaxValue) as SqlCodeExpr;
		//		parser.Scanner.Consume(SqlToken.Equal);
		//		var valueExpr = parser.ParseExpIgnoreComment();
		//		items.Add(new AssignSqlCodeExpr
		//		{
		//			Left = name,
		//			Right = valueExpr,
		//		});
		//	} while (parser.Scanner.Match(SqlToken.Comma));
		//	return items;
		//}
	}
}