using PreviewLibrary.Exceptions;

namespace PreviewLibrary
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