using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SetDealockPrioritySqlCodeExpr : SqlCodeExpr
	{
		public string Priority { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"SET DEADLOCK_PRIORITY {Priority}");
		}
	}
}