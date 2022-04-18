using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql
{
	public class TopSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr NumberExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("TOP ");
			NumberExpr.WriteToStream(stream);
		}
	}
}