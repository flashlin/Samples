using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class IfExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public SqlExpr Body { get; set; }
		public SqlExprList ElseIfList { get; set; }
		public SqlExpr ElseBody { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"IF {Condition}");
			sb.Append($"{Body}");

			if (ElseIfList != null)
			{
				var elseIfList = $"{ElseIfList}";
				if (!string.IsNullOrEmpty(elseIfList))
				{
					sb.Append($"\r\n{elseIfList}");
				}
			}

			if (ElseBody != null)
			{
				sb.Append($"\r\nELSE {ElseBody}");
			}
			return sb.ToString();
		}
	}
}