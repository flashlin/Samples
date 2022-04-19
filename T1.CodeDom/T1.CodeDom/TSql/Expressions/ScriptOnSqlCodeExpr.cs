using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ScriptOnSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(":ON ERROR EXIT");
		}
	}
}