using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary
{
	public class LineChInfo
	{
		public int LineNumber { get; set; }
		public int ChNumber { get; set; }
		public string Line { get; set; }
		public string[] PrevLines { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine(string.Join("\r\n", PrevLines));
			sb.Append($"Ln:{LineNumber} Ch:{ChNumber} '{Line}'");
			return sb.ToString();
		}
	}
}