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
					return token;
				}
			}
			throw new ScanException();
		}

		public static string PeekString(this IScanner scanner)
		{
			var token = scanner.Peek();
			return scanner.GetSpanString(token);
		}
	}
}
