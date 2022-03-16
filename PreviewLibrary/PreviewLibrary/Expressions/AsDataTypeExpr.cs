using PreviewLibrary.Exceptions;

namespace PreviewLibrary.Expressions
{
	public class AsDataTypeExpr : SqlExpr
	{
		public SqlExpr Object { get; set; }
		public SqlExpr DataType { get; set; }

		public override string ToString()
		{
			return $"{Object} AS {DataType}";
		}
	}
}