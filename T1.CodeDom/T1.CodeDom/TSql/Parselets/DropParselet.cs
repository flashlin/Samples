using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DropParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if(parser.Scanner.IsToken(SqlToken.Table))
			{
				return DropTable(parser);
			}

			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException(helpMessage);
		}

		private DropSqlCodeExpr DropTable(IParser parser)
		{
			parser.Scanner.ConsumeString(SqlToken.Table);
			parser.TryConsume(SqlToken.TempTable, out var tmpTable);
			
			return new DropSqlCodeExpr
			{
				TargetId = "TABLE",
				ObjectId = tmpTable
			};
		}
	}
}