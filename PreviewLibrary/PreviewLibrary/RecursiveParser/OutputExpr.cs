using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class OutputExpr : SqlExpr
	{
		public SqlExprList ColumnList { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"OUTPUT {ColumnList}");
			return sb.ToString();
		}
	}
}