using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class BeginTransactionSqlCodeExpr : SqlCodeExpr
	{

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("BEGIN TRANSACTION");
		}
	}
}