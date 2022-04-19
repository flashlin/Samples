using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ScriptSetvarSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(":SETVAR ");
			Name.WriteToStream(stream);
			stream.Write(" ");
			Value.WriteToStream(stream);
		}
	}
}