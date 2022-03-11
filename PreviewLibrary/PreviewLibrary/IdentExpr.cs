namespace PreviewLibrary
{
	public class IdentExpr : SqlExpr
	{
		public string Name { get; set; }
		public string ObjectId { get; set; }
		public string DatabaseId { get; set; }

		public override string ToString()
		{
			if( !string.IsNullOrEmpty(DatabaseId) )
			{ 
				return $"{DatabaseId}.{ObjectId}.{Name}";
			}
			if( !string.IsNullOrEmpty(ObjectId) )
			{ 
				return $"{ObjectId}.{Name}";
			}
			return $"{Name}";
		}
	}

	public class UseExpr : SqlExpr
	{
		public IdentExpr ObjectId { get; set; }
	}
}