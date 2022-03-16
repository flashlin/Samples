using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class IntegerExpr : SqlExpr
	{
		public int Value { get; set; }
		public override string ToString()
		{
			return $"{Value}";
		}
	}
}