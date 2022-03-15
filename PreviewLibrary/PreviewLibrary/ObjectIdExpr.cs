namespace PreviewLibrary
{
	public class ObjectIdExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }

		public override string ToString()
		{
			return $"OBJECT::{Name}";
		}
	}
}