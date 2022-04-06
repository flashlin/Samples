using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class MultiCommentSqlDom : SqlDom
	{
		public string Content { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Content);
		}
	}
}