using PreviewLibrary.Pratt.Core.Expressions;

namespace PreviewLibrary.Pratt.Core.Parselets
{
	public interface IInfixParselet
	{
		IExpression Parse(IExpression left, TextSpan token, IParser parser);
		int GetPrecedence();
	}
}
