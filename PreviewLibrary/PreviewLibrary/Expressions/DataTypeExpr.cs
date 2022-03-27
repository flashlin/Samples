using PreviewLibrary.Exceptions;

namespace PreviewLibrary.Expressions
{
	public class DataTypeExpr : SqlExpr
	{
		public string DataType { get; set; }
		public DataTypeSizeExpr DataSize { get; set; }
		public MarkPrimaryKeyExpr PrimaryKey { get; set; }

		public override string ToString()
		{
			var primaryKey = PrimaryKey != null ? $" {PrimaryKey}" : string.Empty;
			return $"{DataType}{DataSize}{primaryKey}";
		}
	}
}