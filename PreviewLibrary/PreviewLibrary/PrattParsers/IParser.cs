using PreviewLibrary.Exceptions;
using PreviewLibrary.PrattParsers.Expressions;
using System.Collections.Generic;

using Parselet = System.Func<
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;

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
		bool TryConsume(SqlToken expectToken, out TextSpan token);
		bool TryConsumes(out List<TextSpan> tokenList, params SqlToken[] expectTokens);
		bool TryConsumes(out List<TextSpan> tokenList, params SqlToken[][] expectTokens);
		SqlDom ParseBy(SqlToken expectPrefixToken);
		bool TryParseBy<TSqlDom>(SqlToken expectPrefixToken, out TSqlDom sqlDom) where TSqlDom: SqlDom;
		ParseException CreateParseException(TextSpan currentSpan);
		SqlDom ParseByAny(params SqlToken[] expectPrefixToken);
		SqlDom ParseBy(Parselet parse);
		string Peek();
		TextSpan Consume(SqlToken expectToken);
		bool TryConsume(string expectToken, out TextSpan token);
	}
}
