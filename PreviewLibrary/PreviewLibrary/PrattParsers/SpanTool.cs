using System;

namespace PreviewLibrary.PrattParsers
{
	public static class SpanTool
	{
		public static unsafe ReadOnlySpan<char> Concat(ReadOnlySpan<char> span0, ReadOnlySpan<char> span1)
		{
			fixed (char* pointer = span0)
			{
				return new ReadOnlySpan<char>(pointer, span0.Length + span1.Length);
			}
		}
	}
}
