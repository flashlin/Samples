using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class SetTransactionIsolationLevelExpr : SqlExpr
	{
		public string ActionName { get; set; }

		public override string ToString()
		{
			return $"SET TRANSACTION ISOLATION LEVEL {ActionName.ToUpper()}";
		}
	}
}