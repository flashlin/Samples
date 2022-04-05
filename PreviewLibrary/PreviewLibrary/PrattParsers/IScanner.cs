using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.PrattParsers
{
	public interface IScanner
	{
		ReadOnlySpan<char> Peek();
		ReadOnlySpan<char> Consume(string expect = null);
	}
}
