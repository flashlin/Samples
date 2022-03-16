using PreviewLibrary.Exceptions;

namespace PreviewLibrary.Expressions
{
	public class DataTypeExpr : SqlExpr
	{
		public string DataType { get; set; }
		public DataTypeSizeExpr DataSize { get; set; }

		public override string ToString()
		{
			return $"{DataType}{DataSize}";
		}
	}
}