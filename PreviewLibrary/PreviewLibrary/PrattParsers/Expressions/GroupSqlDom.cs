using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class GroupSqlDom : SqlDom
	{
		public SqlDom Inner { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("( ");
			Inner.WriteToStream(stream);
			stream.Write(" )");
		}
	}
}