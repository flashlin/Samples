using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class IsNullSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr CheckExpr { get; set; }
		public SqlCodeExpr ReplacementValue { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ISNULL(");
			CheckExpr.WriteToStream(stream);
			stream.Write(", ");
			ReplacementValue.WriteToStream(stream);
			stream.Write(")");
		}
	}
}