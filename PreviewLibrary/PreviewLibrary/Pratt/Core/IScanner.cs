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

	public interface IParser
	{
	}

	public interface InfixParselet<TToken, TExpr>
	{
		TExpr parse(IParser parser, TExpr left, TextSpan token);
		int getPrecedence();
	}
}
