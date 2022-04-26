using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class TableDataTypeSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public List<SqlCodeExpr> ExtraList { get; set; }

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
			
			if (ExtraList != null && ExtraList.Count > 0)
			{
				stream.WriteLine(",");
				ExtraList.WriteToStream(stream);
			}
			
			stream.WriteLine();
			stream.Write(")");
		}
	}
}