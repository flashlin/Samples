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
		TextSpan ConsumeTokenType(string expectTokenType);
	}
}
