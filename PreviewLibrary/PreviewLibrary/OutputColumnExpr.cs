using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class OutputColumnExpr : SqlExpr
	{
		public string ActionName { get; set; }
		public IdentExpr Column { get; set; }
		public IdentExpr Alias { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{ActionName}.{Column}");
			if(Alias != null)
			{
				sb.Append($" {Alias}");
			}
			return sb.ToString();
		}
	}
}