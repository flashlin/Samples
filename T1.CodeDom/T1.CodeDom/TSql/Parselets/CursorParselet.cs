using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class CursorParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.For);

			var selectExpr = parser.ParseExpIgnoreComment();

			return new CursorForSqlCodeExpr
			{
				SelectExpr = selectExpr
			};
		}
	}
}