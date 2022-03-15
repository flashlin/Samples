namespace PreviewLibrary
{
	public class WhenThenExpr : SqlExpr
	{
		public SqlExpr When { get; set; }
		public SqlExpr Then { get; set; }
	}
}