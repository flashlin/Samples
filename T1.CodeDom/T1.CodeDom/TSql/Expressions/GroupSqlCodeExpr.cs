using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class GroupSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr InnerExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("( ");
			InnerExpr.WriteToStream(stream);
			stream.Write(" )");
		}
	}
}