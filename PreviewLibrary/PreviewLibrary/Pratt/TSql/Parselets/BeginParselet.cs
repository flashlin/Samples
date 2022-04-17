using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class BeginParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.Scanner.IsToken(SqlToken.TRANSACTION))
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
			parser.Scanner.Consume(SqlToken.TRANSACTION);
			return new BeginTransactionSqlCodeExpr();
		}
	}
}
