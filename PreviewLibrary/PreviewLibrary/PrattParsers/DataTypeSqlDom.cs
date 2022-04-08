using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class DataTypeSqlDom : SqlDom
	{
		public string DataType { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(DataType.ToUpper());
		}
	}
}