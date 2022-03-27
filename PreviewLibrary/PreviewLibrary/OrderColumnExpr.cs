using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class OrderColumnExpr : SqlExpr
	{
		public IdentExpr Column { get; set; }
		public string OrderType { get; set; }

		public override string ToString()
		{
			if (string.IsNullOrEmpty(OrderType))
			{
				return $"{Column}";
			}
			return $"{Column} {OrderType}";
		}
	}
}