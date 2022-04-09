namespace PreviewLibrary.Pratt.Core.Expressions
{
	public class PrefixExpression : IExpression
	{
		public string Token { get; set; }
		public IExpression Right { get; set; }
	}
}
