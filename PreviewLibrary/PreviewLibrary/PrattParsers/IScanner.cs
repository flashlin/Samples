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
		bool MatchIgnoreCase(string expect);
	}
}
