using System.Collections.Generic;

namespace PreviewLibrary
{
	public class UpdateExpr : SqlExpr
	{
		public List<SqlExpr> Fields { get; set; }
		public SqlExpr WhereExpr { get; set; }
	}
}