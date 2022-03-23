using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class NotLikeExpr : SqlExpr
	{
		public SqlExpr Right { get; set; }
		public SqlExpr Left { get; set; }

		public override string ToString()
		{
			return $"{Left} NOT LIKE {Right}";
		}
	}
}