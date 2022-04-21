using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class AssignSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Left { get; set; }
		public string Oper { get; set; } = "=";
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write($" {Oper} ");
			Right.WriteToStream(stream);
		}
	}
}