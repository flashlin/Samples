using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class TableTypeExpr : SqlExpr
	{
		public List<SqlExpr> ColumnTypeList { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine("TABLE (");
			foreach (var columnType in ColumnTypeList)
			{
				if (columnType != ColumnTypeList.First())
				{
					sb.Append("\r\n,");
				}
				sb.Append($"{columnType}");
			}
			sb.Append("\r\n)");
			return sb.ToString();
		}
	}
}