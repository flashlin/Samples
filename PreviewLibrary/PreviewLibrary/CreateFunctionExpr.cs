using System.Collections.Generic;

namespace PreviewLibrary
{
	public class CreateFunctionExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public List<List<ArgumentExpr>> ArgumentsList { get; set; }
		public DataTypeExpr ReturnDataType { get; set; }
		public List<SqlExpr> Body { get; set; }
	}
}