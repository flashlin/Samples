namespace PreviewLibrary
{
	public class ArgumentExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr DataType { get; set; }
		public SqlExpr DefaultValue { get; set; }
	}
}