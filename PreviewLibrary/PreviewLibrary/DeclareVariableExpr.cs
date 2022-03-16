using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class DeclareVariableExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr DataType { get; set; }
	}
}