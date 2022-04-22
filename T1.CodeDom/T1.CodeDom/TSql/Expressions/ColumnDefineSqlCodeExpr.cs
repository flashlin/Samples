using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ColumnDefineSqlCodeExpr : SqlCodeExpr
	{
		public string Name { get; set; }
		public SqlCodeExpr DataType { get; set; }
		public SqlCodeExpr DefaultValue { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Name);
			stream.Write(" ");
			DataType.WriteToStream(stream);
			if (DefaultValue != null)
			{
				stream.Write(" DEFAULT ");
				DefaultValue.WriteToStream(stream);
			}
		}
	}
}