using System;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Exceptions
{
	public class PrecursorException : Exception
	{
		public PrecursorException() : base()
		{
		}

		public PrecursorException(string message) : base(message)
		{
		}

		public PrecursorException(string message, Exception innerException) : base(message, innerException)
		{
		}
	}
}
