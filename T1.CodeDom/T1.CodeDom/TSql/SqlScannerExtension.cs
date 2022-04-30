using System;
using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;

namespace T1.CodeDom.TSql
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

		public static bool MatchAny(this IScanner scanner, params SqlToken[] tokenTypeList)
		{
			scanner.IgnoreComments();
			var expectTokenTypeList = tokenTypeList.Select(x => x.ToString()).ToList();
			var span = scanner.Peek();
			if (!expectTokenTypeList.Contains(span.Type))
			{
				return false;
			}
			scanner.Consume();
			return true;
		}

		public static TextSpan Consume(this IScanner scanner, SqlToken tokenType)
		{
			scanner.IgnoreComments();
			return scanner.Consume<SqlToken>(tokenType);
		}

		public static bool TryConsumeList(this IScanner scanner, out List<TextSpan> spanList, params SqlToken[] tokenTypeList)
		{
			scanner.IgnoreComments();
			var startIndex = scanner.GetOffset();
			spanList = new List<TextSpan>();
			for (var i = 0; i < tokenTypeList.Length; i++)
			{
				var tokenType = tokenTypeList[i];
				if (!scanner.TryConsume<SqlToken>(tokenType, out var span))
				{
					spanList = new List<TextSpan>();
					scanner.SetOffset(startIndex);
					return false;
				}
				spanList.Add(span);
			}
			return true;
		}

		public static bool TryConsumeListAny(this IScanner scanner, out List<TextSpan> spanList, params SqlToken[][] tokenTypeListList)
		{
			for (var i = 0; i < tokenTypeListList.Length; i++)
			{
				var tokenTypeList = tokenTypeListList[i];
				if (scanner.TryConsumeList(out spanList, tokenTypeList))
				{
					return true;
				}
			}
			spanList = new List<TextSpan>();
			return false;
		}

		public static List<TextSpan> ConsumeList(this IScanner scanner, params SqlToken[] tokenTypeList)
		{
			scanner.IgnoreComments();
			var textSpanList = new List<TextSpan>();
			foreach (var tokenType in tokenTypeList)
			{
				textSpanList.Add(scanner.Consume<SqlToken>(tokenType));
			}
			return textSpanList;
		}
	}

	public static class SqlTextSpanExtension
	{
		public static SqlToken GetTokenType(this TextSpan span)
		{
			return (SqlToken)Enum.Parse(typeof(SqlToken), span.Type);
		}
		
		public static bool IsTokenType(this TextSpan span, SqlToken tokenType)
		{
			if (span.IsEmpty)
			{
				return false;
			}
			return span.GetTokenType() == tokenType;
		}
	}
}
