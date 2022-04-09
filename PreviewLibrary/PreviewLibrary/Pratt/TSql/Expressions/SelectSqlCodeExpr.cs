using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class SelectSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SELECT");
		}
	}
}