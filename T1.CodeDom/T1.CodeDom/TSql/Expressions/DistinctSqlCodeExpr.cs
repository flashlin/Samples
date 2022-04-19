using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class DistinctSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("DISTINCT ");
			Value.WriteToStream(stream);
		}
	}
}