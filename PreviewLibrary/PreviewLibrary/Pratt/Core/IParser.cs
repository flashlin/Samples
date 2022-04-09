using System.Collections.Generic;

namespace PreviewLibrary.Pratt.Core
{
	public interface IParser<TExpr>
	{
		TExpr ParseExpression();
		IEnumerable<TExpr> ParseProgram();
	}
}
