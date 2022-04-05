using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class OperatorSqlDom : SqlDom
	{
		public SqlDom Left { get; set; }
		public SqlToken OpType { get; set; }
		public SqlDom Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(OpType.ToString());
			Right.WriteToStream(stream);
		}
	}
}