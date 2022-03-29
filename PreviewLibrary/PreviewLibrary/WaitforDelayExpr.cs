using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class WaitforDelayExpr : SqlExpr
	{
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			return $"WAITFOR DELAY {Value}";
		}
	}
}