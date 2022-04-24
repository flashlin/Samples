using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class BeginParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.Scanner.IsTokenAny(SqlToken.TRANSACTION, SqlToken.TRAN))
			{
				return ParseBeginTransaction(parser);
			}

			parser.Scanner.SetOffset(token.Offset - 1);
			var exprList = parser.ConsumeBeginBody();
			return new BeginSqlCodeExpr()
			{
				Items = exprList,
			};
		}

		private BeginTransactionSqlCodeExpr ParseBeginTransaction(IParser parser)
		{
			parser.Scanner.ConsumeAny(SqlToken.TRANSACTION, SqlToken.TRAN);
			return new BeginTransactionSqlCodeExpr();
		}
	}
}
