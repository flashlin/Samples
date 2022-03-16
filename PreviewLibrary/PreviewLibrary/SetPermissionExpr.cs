using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class SetPermissionExpr : SqlExpr
	{
		public string Permission { get; set; }
		public IdentExpr ToObjectId { get; set; }
		public bool Toggle { get; set; }
	}
}