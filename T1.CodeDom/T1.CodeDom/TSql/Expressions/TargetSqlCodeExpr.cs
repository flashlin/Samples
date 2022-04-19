using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class TargetSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Column { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("TARGET.");
			Column.WriteToStream(stream);
		}
	}
}