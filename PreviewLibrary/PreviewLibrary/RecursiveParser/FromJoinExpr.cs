using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class FromJoinExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public IdentExpr AliasName { get; set; }
		public WithOptionsExpr WithOptions { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Table}");
			if(AliasName != null)
			{
				sb.Append($" AS {AliasName}");
			}
			if(WithOptions != null)
			{
				sb.Append($" {WithOptions}");
			}
			return sb.ToString();
		}
	}
}