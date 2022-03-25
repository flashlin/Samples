using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class ReturnExpr : SqlExpr
	{
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			if (Value == null)
			{
				return string.Empty;
			}
			return $"{Value}";
		}
	}
}