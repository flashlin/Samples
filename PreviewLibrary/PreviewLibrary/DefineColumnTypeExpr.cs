using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class DefineColumnTypeExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public SqlExpr DataType { get; set; }
	}
}