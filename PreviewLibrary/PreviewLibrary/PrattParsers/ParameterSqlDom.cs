using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class ParameterSqlDom : SqlDom
	{
		public VariableSqlDom Name { get; set; }
		public SqlDom DataType { get; set; }
		public SqlDom Size { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			stream.Write(" ");
			DataType.WriteToStream(stream);
			Size?.WriteToStream(stream);
		}
	}
}