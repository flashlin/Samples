using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary
{
	public class CreateSpExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public SqlExprList Arguments { get; set; }
		public List<SqlExpr> Body { get; set; }
	}
}