using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class InsertExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public SqlExprList Fields { get; set; }
		public SqlExprList ValuesList { get; set; }
		public bool IntoToggle { get; set; }

		public override string ToString()
		{
			var intoToken = IntoToggle ? "INTO" : "";
			return $"INSERT {intoToken} {Table} ({Fields}) VALUES {ValuesList}";
		}
	}
}