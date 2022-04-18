using System;
using System.Collections.Generic;
using System.Linq;

namespace T1.CodeDom.Core
{
	public static class IScannerExtension
	{
		public static bool TryConsumeString<TTokenType>(this IScanner scanner, TTokenType tokenType, out string spanText)
			where TTokenType : struct
		{
			if (scanner.TryConsume(tokenType, out var span))
			{
				spanText = scanner.GetSpanString(span);
				return true;
			}
			spanText = string.Empty;
			return false;
		}

		public static TextSpan Consume<TTokenType>(this IScanner scanner, TTokenType expectTokenType)
			where TTokenType : struct
		{
			var startIndex = scanner.GetOffset();
			var token = scanner.ScanNext();
			if (token.IsEmpty)
			{
				scanner.SetOffset(startIndex);
				ThrowHelper.ThrowScanException(scanner, $"Expect scan '{expectTokenType}', but got NONE.");
			}
			if (token.Type != expectTokenType.ToString())
			{
				scanner.SetOffset(startIndex);
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

		public static bool TryConsumeStringAny<TTokenType>(this IScanner scanner, out string token, params TTokenType[] expectTokenTypeList)
			where TTokenType : struct
		{
			var span = scanner.Peek();
			var expectTokenTypeStrList = expectTokenTypeList.Select(x => x.ToString());
			if (!expectTokenTypeStrList.Contains(span.Type))
			{
				token = string.Empty;
				return false;
			}
			scanner.Consume();
			token = scanner.GetSpanString(span);
			return true;
		}

		public static string ConsumeStringAny<TTokenType>(this IScanner scanner, params TTokenType[] expectTokenTypeList)
			where TTokenType : struct
		{
			if (!scanner.TryConsumeStringAny(out var token, expectTokenTypeList))
			{
				ThrowHelper.ThrowScanException(scanner, "");
			}
			return token;
		}

		public static string ConsumeString<TTokenType>(this IScanner scanner, TTokenType expectTokenType)
			where TTokenType : struct
		{
			return scanner.ConsumeStringAny(expectTokenType);
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

		public static bool Match<TTokenType>(this IScanner scanner, TTokenType expectTokenType)
			where TTokenType : struct
		{
			return scanner.TryConsume(expectTokenType, out _);
		}


		public static bool IsTokenAny<TTokenType>(this IScanner scanner, params TTokenType[] tokenTypeList)
			where TTokenType : struct
		{
			var token = scanner.Peek();
			var expectTokenTypeList = tokenTypeList.Select(x => x.ToString());
			return expectTokenTypeList.Contains(token.Type);
		}

		public static bool IsToken<TTokenType>(this IScanner scanner, TTokenType tokenType)
			where TTokenType : struct
		{
			var token = scanner.Peek();
			return token.Type == tokenType.ToString();
		}

		public static bool IsTokenList<TTokenType>(this IScanner scanner, params TTokenType[] tokenTypeList)
			where TTokenType : struct
		{
			for (var i = 0; i < tokenTypeList.Length; i++)
			{
				var tokenType = tokenTypeList[i];
				var token = scanner.Peek(i);
				if (token.Type != tokenType.ToString())
				{
					return false;
				}
			}
			return true;
		}

		public static string PeekString(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetSpanString(token);
		}

		public static bool TryConsume<TTokenType>(this IScanner scanner, TTokenType expectTokenType, out TextSpan tokenSpan)
			where TTokenType : struct
		{
			return scanner.TryConsumeAny(out tokenSpan, expectTokenType);
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
	}
}
