using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class InsertFromSelectExpr : SqlExpr
	{
		public bool IntoToggle { get; set; }
		public SqlExpr Table { get; set; }
		public OutputExpr OutputExpr { get; set; }
		public IntoExpr IntoExpr { get; set; }
		public SelectExpr FromSelect { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append("INSERT");
			if (IntoToggle)
			{
				sb.Append(" INTO");
			}
			sb.Append($" {Table}");

			if (OutputExpr != null)
			{
				sb.AppendLine();
				sb.Append($"{OutputExpr}");
			}

			if (IntoExpr != null)
			{
				sb.AppendLine();
				sb.Append($"{IntoExpr}");
			}


			if (FromSelect != null)
			{
				sb.Append($" {FromSelect}");
			}
			return sb.ToString();
		}
	}
}