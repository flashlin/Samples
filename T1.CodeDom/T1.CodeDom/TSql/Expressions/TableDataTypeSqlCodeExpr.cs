using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class TableDataTypeSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.WriteLine("TABLE");

			if (Name != null)
			{
				stream.Write(" ");
				Name.WriteToStream(stream);
			}
			
			stream.WriteLine("(");
			Columns.WriteToStreamWithComma(stream);
			stream.WriteLine();
			stream.Write(")");
		}
	}
}