using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class InsertIntoFromSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public List<string> ColumnsList { get; set; }
		public List<SqlCodeExpr> OutputList { get; set; }
		public SqlCodeExpr OutputIntoExpr { get; set; }
		public SqlCodeExpr SelectFromExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INSERT INTO ");
			Table.WriteToStream(stream);

			if (ColumnsList != null && ColumnsList.Count > 0)
			{
				stream.Write("(");
				ColumnsList.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			if(OutputList != null && OutputList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("OUTPUT ");
				OutputList.WriteToStreamWithComma(stream);
			}

			if(OutputIntoExpr != null)
			{
				stream.WriteLine();
				OutputIntoExpr.WriteToStream(stream);
			}

			stream.WriteLine();
			SelectFromExpr.WriteToStream(stream);
		}
	}
}