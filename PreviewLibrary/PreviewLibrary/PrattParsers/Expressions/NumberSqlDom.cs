using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class NumberSqlDom : SqlDom
	{
		public string Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Value);
		}
	}
}