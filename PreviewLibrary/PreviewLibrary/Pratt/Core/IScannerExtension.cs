using System.Linq;

namespace PreviewLibrary.Pratt.Core
{
	public static class IScannerExtension
	{
		public static string PeekString(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetSpanString(token);
		}
	}
}
