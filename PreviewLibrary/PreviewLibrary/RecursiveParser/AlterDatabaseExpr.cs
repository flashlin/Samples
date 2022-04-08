using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class AlterDatabaseExpr : SqlExpr
	{
		public IdentExpr DbName { get; set; }
		public string ActionName { get; set; }
		public IdentExpr FileGroupName { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"ALTER DATABASE {DbName}");
			sb.Append($"{ActionName} FILEGROUP {FileGroupName}");
			return sb.ToString();
		}
	}
}