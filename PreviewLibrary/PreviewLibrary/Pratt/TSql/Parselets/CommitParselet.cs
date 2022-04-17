using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CommitParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.TryConsumeAny(out _, SqlToken.TRAN, SqlToken.TRANSACTION);
			return new CommitSqlCodeExpr();
		}
	}
}