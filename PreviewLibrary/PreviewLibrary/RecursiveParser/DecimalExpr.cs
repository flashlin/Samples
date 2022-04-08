using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class DecimalExpr : SqlExpr
	{
		public decimal Value { get; set; }

		public override string ToString()
		{
			return $"{Value}";
		}
	}
}