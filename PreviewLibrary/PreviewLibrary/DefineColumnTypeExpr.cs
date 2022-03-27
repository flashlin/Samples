using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class DefineColumnTypeExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public SqlExpr DataType { get; set; }

		public override string ToString()
		{
			if(DataType == null)
			{
				return $"{Name}";
			}
			return $"{Name} {DataType}";
		}
	}
}