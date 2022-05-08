using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class SortSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public string SortToken { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			if (!string.IsNullOrEmpty(SortToken))
			{
				stream.Write($" {SortToken.ToUpper()}");
			}
		}
	}
}