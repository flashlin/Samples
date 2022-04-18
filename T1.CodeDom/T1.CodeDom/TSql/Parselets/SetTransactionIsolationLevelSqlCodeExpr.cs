using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class SetTransactionIsolationLevelSqlCodeExpr : SqlCodeExpr
	{
		public string Option { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"SET TRANSACTION ISOLATION LEVEL {Option}");
		}
	}
}