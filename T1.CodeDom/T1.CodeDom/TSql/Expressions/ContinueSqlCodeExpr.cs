using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ContinueSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.WriteLine("CONTINUE");
		}
	}
}