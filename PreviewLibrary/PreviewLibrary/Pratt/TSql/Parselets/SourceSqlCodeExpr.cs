using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SourceSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Column { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SOURCE.");
			Column.WriteToStream(stream);
		}
	}
}