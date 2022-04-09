using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
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