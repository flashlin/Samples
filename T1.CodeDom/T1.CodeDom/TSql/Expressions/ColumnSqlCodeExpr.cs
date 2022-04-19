using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ColumnSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr AliasName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			//if (!string.IsNullOrEmpty(AliasName))
			//{
			//	stream.Write($" AS {AliasName}");
			//}

			if (AliasName != null)
			{
				stream.Write(" AS ");
				AliasName.WriteToStream(stream);
			}
		}
	}
}