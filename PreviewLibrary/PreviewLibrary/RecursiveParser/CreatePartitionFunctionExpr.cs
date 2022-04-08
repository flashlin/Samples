using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class CreatePartitionFunctionExpr : SqlExpr
	{
		public IdentExpr FuncName { get; set; }
		public SqlExpr InputParameterType { get; set; }
		public SqlExprList BoundaryValueList { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append("CREATE PARTITION FUNCTION");
			sb.AppendLine($" {FuncName}( {InputParameterType} )");
			sb.AppendLine("AS RANGE");
			sb.AppendLine("FOR VALUES (");
			sb.AppendLine($"{BoundaryValueList}");
			sb.Append(")");
			return sb.ToString();
		}
	}
}