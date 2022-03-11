namespace PreviewLibrary
{

	public class PrintExpr : SqlExpr
	{
		public SqlExpr Content { get; set; }

		public override string ToString()
		{
			return $"PRINT {Content}";
		}
	}
}