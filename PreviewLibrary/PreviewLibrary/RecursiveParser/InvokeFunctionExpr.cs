using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class InvokeFunctionExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public SqlExprList ArgumentsList { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Name}(");
			if(ArgumentsList != null && ArgumentsList.Items.Count > 0)
			{
				sb.Append($" {ArgumentsList} ");
			}
			sb.Append(")");
			return sb.ToString();
		}
	}
}