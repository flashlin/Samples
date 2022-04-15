using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class JoinTableSqlCodeExpr : SqlCodeExpr
	{
		public string JoinType { get; set; }
		public string OuterType { get; set; }
		public SqlCodeExpr SecondTable { get; set; }
		public SqlCodeExpr JoinOnExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"{JoinType.ToUpper()} ");
			if (!string.IsNullOrEmpty(OuterType))
			{
				stream.Write($"{OuterType.ToUpper()} ");
			}
			stream.Write("JOIN ");
			SecondTable.WriteToStream(stream);
			if (JoinOnExpr != null)
			{
				stream.Write(" ");
				JoinOnExpr.WriteToStream(stream);
			}
		}
	}
}