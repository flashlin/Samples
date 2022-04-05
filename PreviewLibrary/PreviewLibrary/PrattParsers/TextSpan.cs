using System;

namespace PreviewLibrary.PrattParsers
{
	public struct TextSpan
	{
		public static TextSpan Empty = new TextSpan
		{
			Type = SqlToken.None,
			Offset = -1,
			Length = 0,
		};

		public SqlToken Type;
		
		public int Offset;
		
		public int Length;
		
		public bool IsEmpty
		{
			get
			{
				return Offset == -1 && Length == 0;
			}
		}

		public char GetCh(ReadOnlySpan<char> textSpan, int index)
		{
			return textSpan.Slice(Offset, Length)[index];
		}

		public string GetString(ReadOnlySpan<char> textSpan)
		{
			if (IsEmpty)
			{
				return String.Empty;
			}
			return textSpan.Slice(Offset, Length).ToString();
		}
	}
}
