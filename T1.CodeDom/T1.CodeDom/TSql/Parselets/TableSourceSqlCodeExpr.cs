using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class TableSourceSqlCodeExpr : SqlCodeExpr
	{
		public TempTableSqlCodeExpr Table { get; set; }
		public string Column { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Table.WriteToStream(stream);
			stream.Write(".");
			stream.Write(Column);
		}
	}
}