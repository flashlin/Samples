using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Pratt.Core
{
	public interface IScanner
	{
		TextSpan Peek();
		TextSpan Consume(string expect = null);
		string GetSpanString(TextSpan span);
		int GetOffset();
		void SetOffset(int offset);
		string GetHelpMessage(TextSpan currentSpan);
	}

	public interface InfixParselet<TExpr>
	{
		TExpr Parse(TExpr left, TextSpan token, IParser<TExpr> parser);
		int GetPrecedence();
	}

	public interface PrefixParselet<TExpr>
	{
		TExpr Parse(TextSpan token, IParser<TExpr> parser);
	}
}
