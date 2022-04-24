using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class DeletedParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.ConsumeToken(SqlToken.Dot);
			var columnExpr = parser.Consume(SqlToken.Identifier);

			return new DeletedColumnSqlCodeExpr
			{
				ColumnExpr = columnExpr
			};
		}
	}

	public class DeletedColumnSqlCodeExpr : SqlCodeExpr 
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DELETED.");
			ColumnExpr.WriteToStream(stream);
		}

		public SqlCodeExpr ColumnExpr { get; set; }
	}

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

	public class BreakParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.ConsumeToken(SqlToken.Semicolon);
			return new BreakSqlCodeExpr();
		}
	}

	public class BreakSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("BREAK;");	
		}
	}
}
