using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class SetVariableSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public string Oper { get; set; }
		public SqlCodeExpr Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SET ");
			Name.WriteToStream(stream);
			stream.Write($" {Oper} ");
			Value.WriteToStream(stream);
		}
	}
}