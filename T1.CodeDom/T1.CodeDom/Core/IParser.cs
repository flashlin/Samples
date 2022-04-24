using System.Collections.Generic;

namespace T1.CodeDom.Core
{
	public interface IParser
	{
		IScanner Scanner { get; }
		IEnumerable<IExpression> ParseProgram();
		IExpression ParseExp(int ctxPrecedence = 0);
		IExpression GetParseExp(int ctxPrecedence = 0);
		IExpression PrefixParse(TextSpan prefixToken, int ctxPrecedence = 0);
		IExpression PrefixParse(IExpression left, int ctxPrecedence = 0);
		bool TryGetPrefixParselet(out IPrefixParselet parselet, TextSpan token);

		void StashInfixParselet<TTokenType>(TTokenType tokenType)
			where TTokenType : struct;

		void UnStashInfixParselet();
	}
}
