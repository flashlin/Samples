namespace PreviewLibrary
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