using PreviewLibrary.Pratt.Core.Expressions;

namespace PreviewLibrary.Pratt.Core.Parselets
{
	public interface IPrefixParselet
	{
		IExpression Parse(TextSpan token, IParser parser);
	}
}
