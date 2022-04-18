using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class BetweenSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr LeftExpr { get; set; }
		public SqlCodeExpr StartExpr { get; set; }
		public SqlCodeExpr EndExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			LeftExpr.WriteToStream(stream);
			stream.Write(" BETWEEN ");
			StartExpr.WriteToStream(stream);
			stream.Write(" AND ");
			EndExpr.WriteToStream(stream);
		}
	}
}