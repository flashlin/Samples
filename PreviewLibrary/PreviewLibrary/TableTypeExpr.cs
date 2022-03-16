using System.Collections.Generic;

namespace PreviewLibrary
{
	public class TableTypeExpr : SqlExpr
	{
		public List<SqlExpr> ColumnTypeList { get; set; }
	}
}