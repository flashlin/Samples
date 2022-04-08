using PreviewLibrary.Exceptions;

namespace PreviewLibrary
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