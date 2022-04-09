using System.Linq;

namespace PreviewLibrary.Pratt.Core
{
	public static class IScannerExtension
	{
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

		public static string PeekString(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetSpanString(token);
		}
	}
}
