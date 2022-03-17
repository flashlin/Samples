using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Expressions
{
	public class ArgumentExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr DataType { get; set; }
		public SqlExpr DefaultValue { get; set; }

		public override string ToString()
		{
			var defaultValue = string.Empty;
			if (DefaultValue != null)
			{
				defaultValue = $"={DefaultValue}";
			}
			return $"{Name} {DataType}{defaultValue}";
		}
	}

	public class SqlExprList : SqlExpr
	{
		public List<SqlExpr> Items { get; set; }

		public override string ToString()
		{
			return string.Join(",", Items.Select(x => $"{x}"));
		}
	}
}