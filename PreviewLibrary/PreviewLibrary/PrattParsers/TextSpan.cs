using System;

namespace PreviewLibrary.PrattParsers
{
	public struct TextSpan
	{
		public static TextSpan Empty = new TextSpan
		{
			Offset = -1,
			Length = 0,
		};
		
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
			return textSpan.Slice(Offset, Length).ToString();
		}
	}
}
