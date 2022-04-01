using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class WithAsExpr : SqlExpr
	{
		public SqlExprList AliasExprList { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"WITH {AliasExprList}");
			return sb.ToString();
		}
	}
}