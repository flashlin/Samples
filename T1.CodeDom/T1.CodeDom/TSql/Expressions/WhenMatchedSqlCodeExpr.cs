using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class WhenMatchedSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr ClauseSearchCondition { get; set; }
		public SqlCodeExpr MergeMatched { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WHEN MATCHED");

			if (ClauseSearchCondition != null)
			{
				stream.Write(" AND ");
				ClauseSearchCondition.WriteToStream(stream);
			}

			stream.WriteLine();
			stream.WriteLine("THEN");
			stream.Indent++;
			MergeMatched.WriteToStream(stream);
			stream.Indent--;
		}
	}
}