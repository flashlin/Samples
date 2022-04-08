using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class TopExpr : SqlExpr
	{
		public SqlExpr Count { get; set; }
		public bool HasParentheses { get; set; }

		public override string ToString()
		{
			if (HasParentheses)
			{
				return $"TOP({Count})";
			}
			return $"TOP {Count}";
		}
	}
}