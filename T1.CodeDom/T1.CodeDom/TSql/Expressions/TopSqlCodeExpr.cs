using T1.CodeDom.Core;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class TopSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr NumberExpr { get; set; }
		public bool IsPercent { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("TOP ");
			NumberExpr.WriteToStream(stream);
			if (IsPercent)
			{
				stream.Write(" PERCENT");
			}
		}
	}
}