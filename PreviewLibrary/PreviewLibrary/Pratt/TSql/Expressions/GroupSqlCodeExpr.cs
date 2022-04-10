using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class GroupSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr InnerExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("( ");
			InnerExpr.WriteToStream(stream);
			stream.Write(" )");
		}
	}
}