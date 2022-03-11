namespace PreviewLibrary
{
	public class SpParameterExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr value { get; set; }
	}
}