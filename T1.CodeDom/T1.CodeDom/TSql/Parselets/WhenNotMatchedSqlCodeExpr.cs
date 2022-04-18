using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class WhenNotMatchedSqlCodeExpr : SqlCodeExpr
	{
		public bool ByTarget { get; set; }
		public SqlCodeExpr ClauseSearchCondition { get; set; }
		public SqlCodeExpr MergeNotMatched { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WHEN NOT MATCHED");

			if (ByTarget)
			{
				stream.Write(" BY TARGET");
			}

			if (ClauseSearchCondition != null)
			{
				stream.Write(" AND ");
				ClauseSearchCondition.WriteToStream(stream);
			}

			stream.WriteLine();
			stream.WriteLine("THEN");
			stream.Indent++;
			MergeNotMatched.WriteToStream(stream);
			stream.Indent--;
		}
	}
}