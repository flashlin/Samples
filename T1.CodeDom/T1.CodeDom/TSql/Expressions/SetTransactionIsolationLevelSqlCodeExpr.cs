using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
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