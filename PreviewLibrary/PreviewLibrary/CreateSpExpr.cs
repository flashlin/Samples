using System.Collections.Generic;

namespace PreviewLibrary
{
	public class CreateSpExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public List<ArgumentExpr> Arguments { get; set; }
		public List<SqlExpr> Body { get; set; }
	}
}