using System.Collections.Generic;

namespace PreviewLibrary.Pratt.Core
{
	public interface IParser
	{
		IScanner Scanner { get; }

		IEnumerable<IExpression> ParseProgram();
		IExpression ParseExp(int ctxPrecedence);
	}
}
