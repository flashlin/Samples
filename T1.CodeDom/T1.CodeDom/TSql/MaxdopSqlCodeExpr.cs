using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql
{
	public class MaxdopSqlCodeExpr : SqlCodeExpr
	{
		public int NumberOfCpu { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"MAXDOP {NumberOfCpu}");
		}
	}
}