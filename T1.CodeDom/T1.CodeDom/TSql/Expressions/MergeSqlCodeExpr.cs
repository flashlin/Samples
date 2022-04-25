using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class MergeSqlCodeExpr : SqlCodeExpr
	{
		public string IntoToken { get; set; }
		public SqlCodeExpr TargetTable { get; set; }
		public List<string> WithOptions { get; set; }
		public SqlCodeExpr TargetTableAliasName { get; set; }
		public SqlCodeExpr TableSource { get; set; }
		public List<SqlCodeExpr> SourceColumnList { get; set; }
		public SqlCodeExpr TableSourceAliasName { get; set; }
		public SqlCodeExpr OnMergeSearchCondition { get; set; }
		public List<SqlCodeExpr> WhenList { get; set; }
		public List<SqlCodeExpr> OutputList { get; set; }

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

			if (SourceColumnList != null && SourceColumnList.Count > 0)
			{
				stream.Write("(");
				SourceColumnList.WriteToStreamWithComma(stream);
				stream.Write(")");
			}

			stream.Write(" ON ");
			OnMergeSearchCondition.WriteToStream(stream);

			if (WhenList != null && WhenList.Count > 0)
			{
				stream.WriteLine();
				WhenList.WriteToStream(stream);
			}

			if (OutputList != null && OutputList.Count > 0)
			{
				stream.WriteLine();
				stream.Write("OUTPUT ");
				OutputList.WriteToStreamWithComma(stream);
			}

			stream.Write(" ;");
		}
	}
}