using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class InSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Left { get; set; }
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(" IN (");
			Right.WriteToStream(stream);
			stream.Write(")");
		}
	}
}