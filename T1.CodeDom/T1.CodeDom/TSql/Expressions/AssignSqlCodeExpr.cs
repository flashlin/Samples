using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class AssignSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Left { get; set; }
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(" = ");
			Right.WriteToStream(stream);
		}
	}
}