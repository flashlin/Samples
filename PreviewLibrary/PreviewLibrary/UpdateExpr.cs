using System.Collections.Generic;

namespace PreviewLibrary
{
	public class UpdateExpr : SqlExpr
	{
		public List<AssignSetExpr> Fields { get; set; }
	}
}