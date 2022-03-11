namespace PreviewLibrary
{
	public class StringExpr : SqlExpr
	{
		public string Text { get; set; }

		public override string ToString()
		{
			return Text;
		}
	}
}