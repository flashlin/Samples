using PreviewLibrary.PrattParsers.Expressions;

namespace PreviewLibrary.PrattParsers
{
	public interface IParser
	{
		SqlDom ParseProgram();
		SqlDom ParseExp(int ctxPrecedence);
		bool Match(string expect);
		void Consume(string expect);
		string GetSpanString(TextSpan span);
		bool Match(SqlToken expectToken);
	}
}
