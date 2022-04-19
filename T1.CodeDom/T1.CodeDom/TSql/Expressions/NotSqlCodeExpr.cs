using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class NotSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("NOT ");
			Right.WriteToStream(stream);
		}
	}
}