﻿using System.Collections.Generic;
using System.Text;

namespace T1.CodeDom.Core
{
	public interface IScanner
	{
		TextSpan Consume(string expect = null);

		string GetHelpMessage(TextSpan currentSpan);

		int GetOffset();

		string GetSpanString(TextSpan span);

		TextSpan Peek(int n = 0);

		void SetOffset(int offset);
		TextSpan ScanNext();
		bool IsSymbol(int n = 0);
		bool IsSymbol(TextSpan span);
		bool IsFuncName(string spanStr);
	}
}