namespace PreviewLibrary
{
	public class ArgumentExpr : SqlExpr
	{
		public string Name { get; set; }
		public DataTypeExpr DataType { get; set; }
		public SqlExpr DefaultValue { get; set; }
	}
}