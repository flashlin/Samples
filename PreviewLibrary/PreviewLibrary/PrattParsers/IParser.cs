using PreviewLibrary.PrattParsers.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.PrattParsers
{
	public interface IParser
	{
		IEnumerable<SqlDom> ParseProgram();
		SqlDom ParseExp(int ctxPrecedence);
		bool Match(string expect);
		TextSpan Consume(string expect="");
		string GetSpanString(TextSpan span);
		bool Match(SqlToken expectToken);
	}
}
