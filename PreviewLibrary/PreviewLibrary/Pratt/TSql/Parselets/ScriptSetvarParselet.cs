using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ScriptSetvarParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var variableName = parser.ConsumeAny(SqlToken.Identifier);
			var value = parser.ConsumeAny(SqlToken.DoubleQuoteString);
			return new ScriptSetvarSqlCodeExpr
			{
				Name = variableName as SqlCodeExpr,
				Value = value as SqlCodeExpr
			};
		}
	}
}
