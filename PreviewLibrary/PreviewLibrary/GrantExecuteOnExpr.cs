namespace PreviewLibrary
{
	public class GrantExecuteOnExpr : SqlExpr
	{
		public IdentExpr ToRoleId { get; set; }
		public IdentExpr AsDbo { get; set; }
		public ObjectIdExpr OnObjectId { get; set; }

		public override string ToString()
		{
			return $"GRANT EXECUTE ON {OnObjectId} TO {ToRoleId} AS {AsDbo}";
		}
	}
}