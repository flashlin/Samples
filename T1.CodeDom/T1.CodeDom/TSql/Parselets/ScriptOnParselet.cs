using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class ScriptOnParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Error);
			parser.Scanner.Consume(SqlToken.Exit);
			return new ScriptOnSqlCodeExpr();
		}
	}
}
