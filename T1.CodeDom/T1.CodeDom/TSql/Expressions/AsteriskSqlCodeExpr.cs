using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class AsteriskSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("*");
		}
	}
}