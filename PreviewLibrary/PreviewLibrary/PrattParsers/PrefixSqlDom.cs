using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class PrefixSqlDom : SqlDom
	{
		public SqlToken ValueType { get; set; }
		public SqlDom Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(ValueType.ToString());
			Value.WriteToStream(stream);
		}
	}
}