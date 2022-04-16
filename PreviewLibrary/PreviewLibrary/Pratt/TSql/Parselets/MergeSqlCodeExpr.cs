using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class MergeSqlCodeExpr : SqlCodeExpr
	{
		public string IntoToken { get; set; }
		public SqlCodeExpr TargetTable { get; set; }
		public List<string> WithOptions { get; set; }
		public SqlCodeExpr TargetTableAliasName { get; set; }
		public SqlCodeExpr TableSource { get; set; }
		public SqlCodeExpr TableSourceAliasName { get; set; }
		public SqlCodeExpr OnMergeSearchCondition { get; set; }
		public WhenMatchedSqlCodeExpr WhenMatched { get; set; }
		public List<SqlCodeExpr> WhenNotMatchedList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("MERGE");

			if (!string.IsNullOrEmpty(IntoToken))
			{
				stream.Write($" {IntoToken}");
			}

			stream.Write(" ");
			TargetTable.WriteToStream(stream);

			if (WithOptions != null && WithOptions.Count > 0)
			{
				stream.Write(" WITH(");
				WithOptions.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			if (TargetTableAliasName != null)
			{
				stream.Write(" AS ");
				TargetTableAliasName.WriteToStream(stream);
			}

			stream.WriteLine();
			stream.Write("USING ");
			TableSource.WriteToStream(stream);

			if (TableSourceAliasName != null)
			{
				stream.Write(" AS ");
				TableSourceAliasName.WriteToStream(stream);
			}

			stream.Write(" ON ");
			OnMergeSearchCondition.WriteToStream(stream);

			if (WhenMatched != null)
			{
				stream.WriteLine();
				WhenMatched.WriteToStream(stream);
			}

			if (WhenNotMatchedList != null && WhenNotMatchedList.Count > 0)
			{
				stream.WriteLine();
				WhenNotMatchedList.WriteToStream(stream);
			}
		}
	}
}