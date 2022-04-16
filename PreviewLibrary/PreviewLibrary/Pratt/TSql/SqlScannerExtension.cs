using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql
{
	public static class SqlScannerExtension
	{
		public static bool TryConsume(this IScanner scanner, SqlToken tokenType, out TextSpan span)
		{
			scanner.IgnoreComments();
			return scanner.TryConsume<SqlToken>(tokenType, out span);
		}

		public static bool Match(this IScanner scanner, SqlToken tokenType)
		{
			scanner.IgnoreComments();
			return scanner.Match<SqlToken>(tokenType);
		}

		public static TextSpan Consume(this IScanner scanner, SqlToken tokenType)
		{
			scanner.IgnoreComments();
			return scanner.Consume<SqlToken>(tokenType);
		}
	}
}
