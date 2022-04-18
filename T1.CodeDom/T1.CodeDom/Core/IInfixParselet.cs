namespace T1.CodeDom.Core
{
	public interface IInfixParselet
	{
		IExpression Parse(IExpression left, TextSpan token, IParser parser);
		int GetPrecedence();
	}
}
