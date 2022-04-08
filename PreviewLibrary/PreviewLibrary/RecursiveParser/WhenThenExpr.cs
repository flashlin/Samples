using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class WhenThenExpr : SqlExpr
	{
		public SqlExpr When { get; set; }
		public SqlExpr Then { get; set; }

		public override string ToString()
		{
			return $"WHEN {When} THEN {Then}";
		}
	}
}