using System;

namespace PreviewLibrary.Pratt
{
	public struct TextSpan<TTokenType>
	{
		public static TextSpan<TTokenType> Empty = new TextSpan<TTokenType>
		{
			Offset = -1,
			Length = 0,
		};

		public TTokenType Type;

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
				return string.Empty;
			}
			return textSpan.Slice(Offset, Length).ToString();
		}

		public TextSpan<TTokenType> Concat(TextSpan<TTokenType> span1)
		{
			return new TextSpan<TTokenType>
			{
				Offset = Offset,
				Length = Length + span1.Length
			};
		}
	}
}
