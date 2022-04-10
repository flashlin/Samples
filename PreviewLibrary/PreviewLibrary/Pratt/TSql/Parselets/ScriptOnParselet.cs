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
			parser.Scanner.ConsumeTokenType(SqlToken.Error.ToString());
			parser.Scanner.ConsumeTokenType(SqlToken.Exit.ToString());
			return new ScriptOnSqlCodeExpr();
		}
	}
}
