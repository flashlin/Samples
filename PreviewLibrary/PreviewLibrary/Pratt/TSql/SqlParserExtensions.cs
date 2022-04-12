using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql
{
	public static class SqlParserExtensions
	{
		public static bool Match(this IParser parser, SqlToken tokenType)
		{
			return parser.MatchTokenType(tokenType.ToString());
		}
	}

	public static class SqlCodeExprExtension
	{
		public static void WriteToStreamWithComma(this IEnumerable<SqlCodeExpr> exprList, IndentStream stream)
		{
			foreach (var expr in exprList.Select((val, idx) => new { val, idx }))
			{
				if (expr.idx != 0)
				{
					stream.Write(", ");
				}
				expr.val.WriteToStream(stream);
			}
		}

		public static void WriteToStreamWithComma(this IEnumerable<string> strList, IndentStream stream)
		{
			foreach (var str in strList.Select((val, idx) => new { val, idx }))
			{
				if (str.idx != 0)
				{
					stream.Write(", ");
				}
				stream.Write(str.val);
			}
		}
	}
}
