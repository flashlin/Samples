using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class ColumnSetExpr : SqlExpr
	{
		public string SetVariableName { get; set; }
		public SqlExpr Column { get; set; }

		public override string ToString()
		{
			return $"{SetVariableName} = {Column}";
		}
	}
}