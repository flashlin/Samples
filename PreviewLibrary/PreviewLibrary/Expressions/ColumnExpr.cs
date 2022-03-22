namespace PreviewLibrary.Exceptions
{
	public class ColumnExpr : SqlExpr
	{
		public string Name { get; set; }
		public string Database { get; set; }
		public string Table { get; set; }
		public string AliasName { get; set; }

		public override string ToString()
		{
			var b3 = string.IsNullOrEmpty(Database);
			var b2 = string.IsNullOrEmpty(Table);

			var aliasName = string.IsNullOrEmpty(AliasName) ? "" : $" as {AliasName}";

			if (b3 && b2)
			{
				return $"{Name}{aliasName}";
			}

			if (b3)
			{
				return $"{Table}.{Name}{aliasName}";
			}

			return $"{Database}.{Table}.{Name}{aliasName}";
		}
	}
}