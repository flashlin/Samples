using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class BeginTransactionSqlCodeExpr : SqlCodeExpr
	{

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("BEGIN TRANSACTION");
		}
	}
}