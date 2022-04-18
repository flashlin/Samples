using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class TruncateTableParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Table);
			var tableName = parser.ConsumeObjectId();
			return new TruncateTableSqlCodeExpr()
			{
				TableName = tableName,
			};
		}
	}
}
