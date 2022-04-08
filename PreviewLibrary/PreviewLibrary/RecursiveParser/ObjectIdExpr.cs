using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
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