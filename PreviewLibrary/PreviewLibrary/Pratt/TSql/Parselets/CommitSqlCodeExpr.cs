using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CommitSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("COMMIT TRANSACTION");
		}
	}
}