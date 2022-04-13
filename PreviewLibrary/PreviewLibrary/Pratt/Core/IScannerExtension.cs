using System;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.Core
{
	public static class IScannerExtension
	{
		public static TextSpan Consume<TTokenType>(this IScanner scanner, TTokenType expectTokenType)
			where TTokenType : struct
		{
			var token = scanner.ScanNext();
			if (token.IsEmpty)
			{
				ThrowHelper.ThrowScanException(scanner, $"Expect scan '{expectTokenType}', but got NONE.");
			}
			if (token.Type != expectTokenType.ToString())
			{
				ThrowHelper.ThrowScanException(scanner, $"Expect scan {expectTokenType}, but got {token.Type}.");
			}
			return token;
		}

		public static TextSpan ConsumeAny<TTokenType>(this IScanner scanner, params TTokenType[] tokenTypes)
			where TTokenType : struct
		{
			for (var i = 0; i < tokenTypes.Length; i++)
			{
				var tokenType = tokenTypes[i].ToString();
				var token = scanner.Peek();
				if (token.Type == tokenType)
				{
					scanner.Consume();
					return token;
				}
			}
			var helpMessage = scanner.GetHelpMessage(scanner.Peek());
			var tokenTypesStr = string.Join(",", tokenTypes.Select(x => x.ToString()));
			throw new ScanException($"Expect one of {tokenTypesStr}.\r\n{helpMessage}");
		}

		public static string PeekString(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetSpanString(token);
		}

		public static bool TryConsume<TTokenType>(this IScanner scanner, TTokenType expectTokenType, out TextSpan tokenSpan)
			where TTokenType : struct
		{
			return TryConsumeAny(scanner, out tokenSpan, expectTokenType);
		}

		public static bool Match<TTokenType>(this IScanner scanner, TTokenType expectTokenType)
			where TTokenType : struct
		{
			return TryConsume(scanner, expectTokenType, out _);
		}

		public static bool TryConsumeAny<TTokenType>(this IScanner scanner, out TextSpan outSpan, params TTokenType[] tokenTypes)
			where TTokenType : struct
		{
			for (var i = 0; i < tokenTypes.Length; i++)
			{
				var tokenType = tokenTypes[i].ToString();
				var token = scanner.Peek();
				if (token.Type == tokenType)
				{
					scanner.Consume();
					outSpan = token;
					return true;
				}
			}
			outSpan = TextSpan.Empty;
			return false;
		}

		public static IEnumerable<string> ConsumeToStringListByDelimiter<TTokenType>(this IScanner scanner,
			TTokenType delimiter, params TTokenType[] tokenTypes)
			where TTokenType : struct
		{
			do
			{
				if (scanner.TryConsumeAny(out var token, tokenTypes))
				{
					var tokenStr = scanner.GetSpanString(token);
					yield return tokenStr;
				}
			} while (scanner.Match(delimiter));
		}

		public static string GetHelpMessage(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetHelpMessage(token);
		}
	}
}
