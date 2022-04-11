using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ScriptOnParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.ConsumeTokenType(SqlToken.Error);
			parser.Scanner.ConsumeTokenType(SqlToken.Exit);
			return new ScriptOnSqlCodeExpr();
		}
	}
}
