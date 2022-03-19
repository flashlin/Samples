using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class SetVariableExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			return $"SET {Name} = {Value}";
		}
	}
}