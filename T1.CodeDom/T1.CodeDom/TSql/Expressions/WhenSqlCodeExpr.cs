using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class WhenSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr ConditionExpr { get; set; }
		public SqlCodeExpr ThenExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WHEN ");
			ConditionExpr.WriteToStream(stream);
			stream.WriteLine();
			stream.Write("THEN ");
			ThenExpr.WriteToStream(stream);
		}
	}
}