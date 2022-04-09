using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class CommentSqlCodeExpr : SqlCodeExpr
	{
		public string Content { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Content);
		}
	}
}