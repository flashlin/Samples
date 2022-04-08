using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class OutputColumnExpr : SqlExpr
	{
		public string ActionName { get; set; }
		public SqlExpr Column { get; set; }
		public IdentExpr Alias { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();

			if (!string.IsNullOrEmpty(ActionName))
			{
				sb.Append($"{ActionName}.");
			}
			sb.Append($"{Column}");

			if (Alias != null)
			{
				sb.Append($" {Alias}");
			}
			return sb.ToString();
		}
	}
}