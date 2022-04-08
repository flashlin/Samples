using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class ReturnExpr : SqlExpr
	{
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			if (Value == null)
			{
				return "RETURN";
			}
			return $"RETURN {Value}";
		}
	}
}