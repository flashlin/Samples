using PreviewLibrary.Exceptions;
using System.Linq;

namespace PreviewLibrary.RecursiveParser
{
	public class SqlFuncExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr[] Arguments { get; set; }

		public override string ToString()
		{
			var args = string.Join(",", Arguments.Select(x => $"{x}"));
			if (args == string.Empty)
			{
				return $"{Name}()";
			}
			return $"{Name}( {args} )";
		}
	}
}