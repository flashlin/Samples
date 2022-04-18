using System;

namespace T1.CodeDom.Core
{
	public class ScanException : Exception
	{
		public ScanException() : base()
		{
		}

		public ScanException(string message) : base(message)
		{
		}

		public ScanException(string message, Exception innerException) : base(message, innerException)
		{
		}
	}
}
