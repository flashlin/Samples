using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class WaitForParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.ConsumeToken(SqlToken.DELAY);
			var timeExpr = parser.ParseExpIgnoreComment();
			return new WaitForSqlCodeExpr()
			{
				TimeExpr = timeExpr
			};
		}
	}

	public class WaitForSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WAITFOR DELAY ");
			TimeExpr.WriteToStream(stream);
		}

		public SqlCodeExpr TimeExpr { get; set; }
	}

	public class GoParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			return new GoSqlCodeExpr();
		}
	}
}
