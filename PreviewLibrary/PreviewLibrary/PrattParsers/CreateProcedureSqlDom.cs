using PreviewLibrary.PrattParsers.Expressions;
using System.Collections.Immutable;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class CreateProcedureSqlDom : SqlDom
	{
		public SqlDom Name { get; set; }
		public ImmutableArray<SqlDom>.Builder Parameters { get; set; }
		public SqlDom Body { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE PROCEDURE ");
			Name.WriteToStream(stream);
			stream.WriteLine();

			stream.Indent++;
			foreach (var parameter in Parameters.Select((value, idx) => new { value, idx }))
			{
				if (parameter.idx != 0)
				{
					stream.WriteLine(",");
				}
				parameter.value.WriteToStream(stream);
			}
			stream.Indent--;

			stream.WriteLine();
			stream.WriteLine("AS BEGIN");
			Body.WriteToStream(stream);
			stream.WriteLine();
			stream.WriteLine("END");
		}
	}
}