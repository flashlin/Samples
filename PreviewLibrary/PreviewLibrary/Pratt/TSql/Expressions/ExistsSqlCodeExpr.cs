using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ExistsSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr InnerExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("EXISTS( ");
			InnerExpr.WriteToStream(stream);
			stream.Write(" )");
		}
	}
}