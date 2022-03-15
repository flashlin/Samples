namespace PreviewLibrary
{
	public class DeclareVariableExpr : SqlExpr
	{
		public string Name { get; set; }
		public DataTypeExpr DataType { get; set; }
	}
}