using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class FetchParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.ConsumeToken(SqlToken.Next);
			parser.ConsumeToken(SqlToken.From);
			
			var cursorName = parser.ConsumeTableName();

			parser.ConsumeToken(SqlToken.Into);
			var variableNameList = new List<SqlCodeExpr>();
			do
			{
				var variableName = parser.Consume(SqlToken.Variable);
				variableNameList.Add(variableName);
			} while (parser.MatchToken(SqlToken.Comma));

			return new FetchSqlCodeExpr
			{
				CursorName = cursorName,
				VariableNameList = variableNameList
			};
		}
	}
}