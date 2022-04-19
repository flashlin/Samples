using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class RollbackSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ROLLBACK TRANSACTION");
		}
	}
}