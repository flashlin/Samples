using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using PreviewLibrary.Pratt.TSql.Parselets;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlParser : PrattParser
	{
		public TSqlParser(IScanner scanner) : base(scanner)
		{
			Prefix(SqlToken.PLUS, Precedence.PREFIX);
		}

		public SqlCodeExpr ParseExpression()
		{
			return (SqlCodeExpr)ParseExp(0);
		}

		protected void Prefix(SqlToken tokenType, Precedence precedence)
		{
			Register((int)tokenType, new SqlPrefixOperatorParselet(precedence));
		}
	}
}
