using System.Linq;

namespace PreviewLibrary.Pratt.Core
{
	public static class IScannerExtension
	{
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

		public static bool TryConsume<TTokenType>(this IScanner scanner, TTokenType expectTokenType, out TextSpan tokenSpan)
			where TTokenType : struct
		{
			return TryConsumeAny(scanner, out tokenSpan, expectTokenType);
		}

		public static string PeekString(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetSpanString(token);
		}
	}
}
