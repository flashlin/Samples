﻿using System;
using System.Collections.Generic;

namespace T1.CodeDom.Core
{
	public struct TextSpan
	{
		public static TextSpan Empty = new TextSpan
		{
			Type = String.Empty,
			Offset = -1,
			Length = 0,
		};

		public string Type;

		public int Offset;

		public int Length;

		public bool IsEmpty
		{
			get
			{
				return Offset == -1 && Length == 0;
			}
		}

		public List<TextSpan> Comments { get; set; }

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

		public TextSpan Concat(TextSpan span1)
		{
			return new TextSpan
			{
				Offset = Offset,
				Length = Length + span1.Length
			};
		}
	}
}
