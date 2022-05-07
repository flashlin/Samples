using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class InsertSqlCodeExpr : SqlCodeExpr
	{
		public string IntoStr { get; set; }
		public SqlCodeExpr TableName { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public SqlCodeExpr WithExpr { get; set; }
		public List<ExprListSqlCodeExpr> ValuesList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("INSERT ");
			if (!string.IsNullOrEmpty(IntoStr))
			{
				stream.Write($"{IntoStr.ToUpper()} ");
			}
			TableName.WriteToStream(stream);

			if (Columns != null && Columns.Count > 0)
			{
				stream.Write("(");
				Columns.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			if (WithExpr != null)
			{
				stream.Write(" ");
				WithExpr.WriteToStream(stream);
			}

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