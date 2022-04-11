using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;

namespace PreviewLibrary.Pratt.TSql
{
	public static class ThrowHelper
	{
		public static void ThrowParseException(IParser parser, string errorMessage)
		{
			var token = parser.Scanner.Peek();
			var helpMessage = parser.Scanner.GetHelpMessage(token);
			throw new ParseException($"{errorMessage}\r\n{helpMessage}");
		}

		public static void ThrowScanException(IScanner scanner, string errorMessage)
		{
			var token = scanner.Peek();
			var helpMessage = scanner.GetHelpMessage(token);
			throw new ScanException($"{errorMessage}\r\n{helpMessage}");
		}
	}
}
