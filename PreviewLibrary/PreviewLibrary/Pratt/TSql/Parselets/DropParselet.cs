using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
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