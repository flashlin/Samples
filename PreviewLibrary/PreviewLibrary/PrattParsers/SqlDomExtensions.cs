using PreviewLibrary.PrattParsers.Expressions;
using System.Collections.Immutable;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public static class SqlDomExtensions
	{
		public static void WriteToStream(this ImmutableArray<SqlDom>.Builder array, IndentStream stream, string comma=", ")
		{
			foreach (var item in array.Select((value, index) => new { value, index }))
			{
				if (item.index != 0)
				{
					stream.Write(comma);
				}
				item.value.WriteToStream(stream);
			}
		}
	}
}