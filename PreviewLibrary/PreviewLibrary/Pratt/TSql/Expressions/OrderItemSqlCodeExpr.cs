using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class OrderItemSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public string AscOrDesc { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			if (!string.IsNullOrEmpty(AscOrDesc))
			{
				stream.Write($" {AscOrDesc}");
			}
		}
	}
}