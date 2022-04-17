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
	}
}