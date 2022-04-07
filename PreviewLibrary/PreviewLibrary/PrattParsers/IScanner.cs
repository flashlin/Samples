using System.Buffers;
using System.Text;

namespace PreviewLibrary.PrattParsers
{
	public interface IScanner
	{
		TextSpan Peek();
		TextSpan Consume(string expect = null);
		string GetSpanString(TextSpan span);
		bool Match(string expect);
		//bool MatchIgnoreCase(string expect);
		bool Match(SqlToken expectToken);
		string GetHelpMessage(TextSpan currentSpan);
		int GetOffset();
		void SetOffset(int offset);
	}
}
