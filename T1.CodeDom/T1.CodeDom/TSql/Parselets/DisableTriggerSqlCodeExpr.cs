using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class DisableTriggerSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr TriggerName { get; set; }
		public SqlCodeExpr ObjectExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DISABLE TRIGGER ");
			TriggerName.WriteToStream(stream);
			stream.Write(" ON ");
			ObjectExpr.WriteToStream(stream);
		}
	}
}