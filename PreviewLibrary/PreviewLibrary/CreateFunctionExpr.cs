using System.Collections.Generic;

namespace PreviewLibrary
{
	public class CreateFunctionExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public List<List<ArgumentExpr>> ArgumentsList { get; set; }
		public SqlExpr ReturnDataType { get; set; }
		public List<SqlExpr> Body { get; set; }
	}
}