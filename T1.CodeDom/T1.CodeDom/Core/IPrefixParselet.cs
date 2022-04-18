namespace T1.CodeDom.Core
{
	public interface IPrefixParselet
	{
		IExpression Parse(TextSpan token, IParser parser);
	}
}
