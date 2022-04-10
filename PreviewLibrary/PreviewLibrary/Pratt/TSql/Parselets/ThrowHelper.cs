using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public static class ThrowHelper
	{
		public static void ThrowParseException(IParser parser, string errorMessage)
		{
			var token = parser.Scanner.Peek();
			var helpMessage = parser.Scanner.GetHelpMessage(token);
			throw new ParseException($"{errorMessage}\r\n{helpMessage}");
		}
	}
}
