using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Pratt
{
	public interface IScanner<TTokenType>
	{
		TextSpan<TTokenType> Peek();
		TextSpan<TTokenType> Consume(string expect=null);
		string GetSpanString(TextSpan<TTokenType> span);
		int GetOffset();
		void SetOffset(int offset);
		string GetHelpMessage(TextSpan<TTokenType> currentSpan);
	}

	public interface IParser
	{
	}

	public interface InfixParselet<TToken, TExpr>
	{
		TExpr parse(IParser parser, TExpr left, TextSpan<TToken> token);
		int getPrecedence();
	}
}
