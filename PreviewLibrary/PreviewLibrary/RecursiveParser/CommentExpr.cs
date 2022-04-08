using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class CommentExpr : SqlExpr
	{
		public string Text { get; set; }

		public override string ToString()
		{
			return Text;
		}
	}
}