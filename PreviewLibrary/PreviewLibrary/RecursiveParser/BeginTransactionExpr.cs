using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class BeginTransactionExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"BEGIN TRANSACTION";
		}
	}
	
	public class RollbackTransactionExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"ROLLBACK TRANSACTION";
		}
	}
}