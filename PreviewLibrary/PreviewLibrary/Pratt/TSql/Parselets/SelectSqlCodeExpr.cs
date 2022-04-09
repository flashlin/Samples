using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SelectSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SELECT");
		}
	}
}