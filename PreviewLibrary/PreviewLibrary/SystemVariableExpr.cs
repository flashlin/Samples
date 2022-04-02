using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class SystemVariableExpr : SqlExpr
	{
		public string Name { get; set; }

		public override string ToString()
		{
			return $"{Name}";
		}
	}
}