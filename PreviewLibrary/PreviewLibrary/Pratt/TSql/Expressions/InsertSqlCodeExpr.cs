using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class InsertSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr TableName { get; set; }
		public List<string> Columns { get; set; }
		public List<ExprListSqlCodeExpr> ValuesList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INSERT ");
			TableName.WriteToStream(stream);
			
			stream.Write("(");
			Columns.WriteToStreamWithComma(stream);
			stream.Write(")");
			stream.Write(" VALUES");
			stream.WriteLine();
			foreach (var values in ValuesList.Select((val, idx) => new { val, idx }))
			{
				if (values.idx != 0)
				{
					stream.WriteLine(",");
				}
				stream.Write("(");
				values.val.WriteToStream(stream);
				stream.Write(")");
			}
		}
	}
}