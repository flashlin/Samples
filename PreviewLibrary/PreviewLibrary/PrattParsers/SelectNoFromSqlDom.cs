using PreviewLibrary.PrattParsers.Expressions;
using System.Collections.Immutable;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class SelectNoFromSqlDom : SqlDom
	{
		public ImmutableArray<SqlDom>.Builder Columns { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SELECT ");
			foreach (var column in Columns.Select((value, index) => new { value, index }))
			{
				if (column.index != 0)
				{
					stream.Write(", ");
				}
				column.value.WriteToStream(stream);
			}
		}
	}
}