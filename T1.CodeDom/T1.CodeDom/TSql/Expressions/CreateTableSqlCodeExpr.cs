using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class CreateTableSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> ColumnsList { get; set; }
		public bool IsSemicolon { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE TABLE ");
			Name.WriteToStream(stream);
			stream.WriteLine("(");
			ColumnsList.WriteToStreamWithCommaLine(stream);
			stream.WriteLine();
			stream.Write(")");
			if (IsSemicolon)
			{
				stream.Write(" ;");
			}
		}
	}
}