using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class IsNullSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr CheckExpr { get; set; }
		public SqlCodeExpr ReplacementValue { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ISNULL(");
			CheckExpr.WriteToStream(stream);
			stream.Write(", ");
			ReplacementValue.WriteToStream(stream);
			stream.Write(")");
		}
	}
}