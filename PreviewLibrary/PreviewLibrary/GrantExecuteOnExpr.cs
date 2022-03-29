using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class GrantExecuteOnExpr : SqlExpr
	{
		public string ExecAction { get; set; }
		public IdentExpr ToRoleId { get; set; }
		public IdentExpr AsDbo { get; set; }
		public SqlExpr OnObjectId { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"GRANT {ExecAction} ON {OnObjectId} TO {ToRoleId}");
			if( AsDbo != null)
			{
				sb.Append($" AS {AsDbo}");
			}
			return sb.ToString();
		}
	}
}