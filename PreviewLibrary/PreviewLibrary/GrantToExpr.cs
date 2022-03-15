namespace PreviewLibrary
{
	public class GrantToExpr : SqlExpr
	{
		public string Permission { get; set; }
		public IdentExpr ToObjectId { get; set; }

		public override string ToString()
		{
			return $"GRANT {Permission} {ToObjectId}";
		}
	}
}