using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class DeleteExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public WithOptionsExpr WithOptions { get; set; }
		public SqlExpr WhereExpr { get; set; }
		public OutputExpr OutputExpr { get; set; }
		public IntoExpr IntoExpr { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"DELETE FROM {Table}");

			if(WithOptions != null)
			{
				sb.Append($" {WithOptions}");
			}

			if(OutputExpr != null)
			{
				sb.AppendLine();
				sb.Append($"{OutputExpr}");
			}

			if(IntoExpr != null)
			{
				sb.AppendLine();
				sb.Append($"{IntoExpr}");
			}

			if(WhereExpr != null)
			{
				sb.Append($" WHERE {WhereExpr}");
			}
			return sb.ToString();
		}
	}
}