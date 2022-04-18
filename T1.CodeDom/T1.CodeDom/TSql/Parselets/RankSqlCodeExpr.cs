using System.Collections.Generic;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class RankSqlCodeExpr : SqlCodeExpr 
	{
		public List<SqlCodeExpr> PartitionColumnList { get; set; }
		public List<SqlCodeExpr> OrderByClause { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("RANK() OVER(");
			stream.WriteLine();
			stream.Indent++;

			if(PartitionColumnList != null && PartitionColumnList.Count>0)
			{
				stream.Write("PARTITION BY ");
				PartitionColumnList.WriteToStreamWithComma(stream);
			}

			stream.WriteLine();
			stream.Write("ORDER BY ");
			OrderByClause.WriteToStreamWithComma(stream);

			stream.Indent--;
			stream.Write(")");
		}
	}
}