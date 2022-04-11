using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Pratt.Core
{
	public interface IScanner
	{
		TextSpan Consume(string expect = null);

		TextSpan Consume<TTokenType>(TTokenType expectTokenType)
			where TTokenType : struct;

		TextSpan ConsumeAny<TTokenType>(params TTokenType[] tokenTypes)
					where TTokenType : struct;
		string GetHelpMessage(TextSpan currentSpan);

		int GetOffset();

		string GetSpanString(TextSpan span);

		TextSpan Peek();
		void SetOffset(int offset);
		bool TryConsumeTokenType<TTokenType>(TTokenType expectTokenType, out TextSpan tokenSpan)
			where TTokenType : struct;
	}
}