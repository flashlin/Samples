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
		string GetTokenTypeName<TTokenType>(int tokenTypeNumber);
	}

	public interface IExpression
	{
	}

	public interface InfixParselet
	{
		IExpression Parse(IExpression left, TextSpan token, IParser parser);
		int GetPrecedence();
	}

	public interface PrefixParselet
	{
		IExpression Parse(TextSpan token, IParser parser);
	}
}
