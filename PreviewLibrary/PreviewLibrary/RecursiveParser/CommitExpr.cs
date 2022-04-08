using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class CommitExpr : SqlExpr
	{
		public string ActionName { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"COMMIT");
			if (!string.IsNullOrEmpty(ActionName))
			{
				sb.Append($" {ActionName}");
			}
			return sb.ToString();
		}
	}
}