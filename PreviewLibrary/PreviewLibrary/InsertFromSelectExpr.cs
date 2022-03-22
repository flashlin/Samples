using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class InsertFromSelectExpr : SqlExpr
	{
		public bool IntoToggle { get; set; }
		public IdentExpr Table { get; set; }
		public SelectExpr FromSelect { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append("INSERT");
			if( IntoToggle)
			{
				sb.Append(" INTO");
			}
			sb.Append($" {Table}");
			if( FromSelect != null)
			{
				sb.Append($" {FromSelect}");
			}
			return sb.ToString();
		}
	}
}