namespace PreviewLibrary
{
	public class DeleteExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public SqlExpr WhereExpr { get; set; }

		public override string ToString()
		{
			var whereStr = "";
			if( WhereExpr != null)
			{
				whereStr = $" {WhereExpr}";
			}
			return $"DELETE {Table}{whereStr}";
		}
	}
}