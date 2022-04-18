using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
