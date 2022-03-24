namespace PreviewLibrary.Exceptions
{
	public class ColumnExpr : SqlExpr
	{
		public SqlExpr Name { get; set; }
		public string AliasName { get; set; }

		public override string ToString()
		{
			var aliasName = string.IsNullOrEmpty(AliasName) ? "" : $" as {AliasName}";
			return $"{Name}{aliasName}";
		}
	}
}