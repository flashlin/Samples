using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class AliasExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public IdentExpr AliasName { get; set; }

		public override string ToString()
		{
			if (AliasName != null)
			{
				return $"{Left} AS {AliasName}";
			}
			return $"{Left}";
		}
	}
}