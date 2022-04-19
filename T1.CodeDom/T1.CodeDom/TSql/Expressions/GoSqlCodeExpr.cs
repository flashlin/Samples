using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class GoSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("GO");
		}
	}
}