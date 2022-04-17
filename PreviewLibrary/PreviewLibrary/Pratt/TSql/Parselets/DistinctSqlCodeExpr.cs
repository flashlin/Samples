using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class DistinctSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DISTINCT ");
			Value.WriteToStream(stream);
		}
	}
}