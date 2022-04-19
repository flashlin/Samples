using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class RollbackParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.TryConsumeAny(out _, SqlToken.TRAN, SqlToken.TRANSACTION);
			return new RollbackSqlCodeExpr();
		}
	}
}
