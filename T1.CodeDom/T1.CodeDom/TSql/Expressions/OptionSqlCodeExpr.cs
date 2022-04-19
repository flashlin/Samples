using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class OptionSqlCodeExpr : SqlCodeExpr
	{
		public MaxdopSqlCodeExpr Maxdop { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("OPTION");
			stream.Write("(");
			Maxdop.WriteToStream(stream);
			stream.Write(")");
		}
	}
}