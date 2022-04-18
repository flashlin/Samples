using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class SetLockTimeoutSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr TimeoutPeriod { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SET LOCK_TIMEOUT ");
			TimeoutPeriod.WriteToStream(stream);
		}
	}
}