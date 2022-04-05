using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class OperatorSqlDom : SqlDom
	{
		public SqlDom Left { get; set; }
		public SqlToken OperType { get; set; }
		public SqlDom Right { get; set; }
		public string Oper { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write($" {Oper} ");
			Right.WriteToStream(stream);
		}
	}
}