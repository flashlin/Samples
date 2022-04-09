using System;

namespace PreviewLibrary.Pratt.TSql
{

	public class TSqlScanner : IScanner<SqlToken>
	{
		public TextSpan<SqlToken> Consume(string expect = null)
		{
			throw new NotImplementedException();
		}

		public string GetHelpMessage(TextSpan<SqlToken> currentSpan)
		{
			throw new NotImplementedException();
		}

		public int GetOffset()
		{
			throw new NotImplementedException();
		}

		public string GetSpanString(TextSpan<SqlToken> span)
		{
			throw new NotImplementedException();
		}

		public TextSpan<SqlToken> Peek()
		{
			throw new NotImplementedException();
		}

		public void SetOffset(int offset)
		{
			throw new NotImplementedException();
		}
	}
}
