using System;

namespace PreviewLibrary.Pratt.TSql
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
