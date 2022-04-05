using System.Collections.Immutable;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class SelectSqlDom : SqlDom
	{
		public ImmutableArray<SqlDom>.Builder Columns { get; set; }
		public SqlDom From { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SELECT ");
			Columns.WriteToStream(stream);
			stream.Write(" FROM ");
			From.WriteToStream(stream);
		}
	}
}