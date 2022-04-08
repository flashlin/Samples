using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class SingleCommentSqlDom : SqlDom
	{
		public string Content { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Content);
			stream.WriteLine();
		}
	}
}