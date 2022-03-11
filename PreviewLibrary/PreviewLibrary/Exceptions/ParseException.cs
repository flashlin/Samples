using System;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Exceptions
{

	public class ParseException : Exception
	{
		public ParseException() : base()
		{
		}

		public ParseException(string message) : base(message)
		{
		}

		public ParseException(string message, Exception innerException) : base(message, innerException)
		{
		}
	}
}
