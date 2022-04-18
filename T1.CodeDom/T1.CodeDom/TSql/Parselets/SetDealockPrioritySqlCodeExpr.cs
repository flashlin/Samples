using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
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