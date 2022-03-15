using System.Collections.Generic;

namespace PreviewLibrary
{
	public class CaseExpr : SqlExpr
	{
		public List<WhenThenExpr> WhenList { get; set; }
		public SqlExpr Else { get; set; }
	}
}