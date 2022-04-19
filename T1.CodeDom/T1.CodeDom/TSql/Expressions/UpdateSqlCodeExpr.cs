using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class UpdateSqlCodeExpr : SqlCodeExpr
	{
		public TopSqlCodeExpr TopCount { get; set; }
		public SqlCodeExpr Table { get; set; }
		public List<string> WithOptions { get; set; }
		public List<SqlCodeExpr> SetColumnsList { get; set; }
		//public SqlCodeExpr FromTable { get; set; }
		//public List<SqlCodeExpr> JoinSelectList { get; set; }
		public List<SqlCodeExpr> FromTableList { get; set; }
		public List<SqlCodeExpr> OutputList { get; set; }
		public SqlCodeExpr WhereExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("UPDATE");

			if (TopCount != null)
			{
				stream.Write(" ");
				TopCount.WriteToStream(stream);
			}

			if (Table != null)
			{
				stream.Write(" ");
				Table.WriteToStream(stream);
			}

			if (WithOptions != null && WithOptions.Count > 0)
			{
				stream.Write(" WITH(");
				WithOptions.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			stream.Write(" SET ");
			SetColumnsList.WriteToStreamWithComma(stream);

			if (FromTableList != null && FromTableList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("FROM ");
				FromTableList.WriteToStreamWithComma(stream);
			}

			//if( FromTable != null)
			//{
			//	stream.WriteLine();
			//	stream.Write("FROM ");
			//	FromTable.WriteToStream(stream);
			//}

			//if( JoinSelectList != null && JoinSelectList.Count > 0)
			//{
			//	stream.WriteLine();
			//	JoinSelectList.WriteToStream(stream);
			//}

			if (OutputList != null && OutputList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("OUTPUT ");
				OutputList.WriteToStreamWithComma(stream);
			}

			if (WhereExpr != null)
			{
				stream.WriteLine();
				stream.Write("WHERE ");
				WhereExpr.WriteToStream(stream);
			}
		}
	}
}