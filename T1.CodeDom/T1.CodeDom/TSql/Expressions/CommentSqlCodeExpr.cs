using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
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