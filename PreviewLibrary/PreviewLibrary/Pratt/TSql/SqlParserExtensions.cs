using PreviewLibrary.Pratt.Core;

namespace PreviewLibrary.Pratt.TSql
{
	public static class SqlParserExtensions
	{
		public static bool Match(this IParser parser, SqlToken tokenType)
		{
			return parser.MatchTokenType(tokenType.ToString());
		}
	}
}
