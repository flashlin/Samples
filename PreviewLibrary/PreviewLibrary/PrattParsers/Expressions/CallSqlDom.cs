using System.Collections.Immutable;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class CallSqlDom : SqlDom
	{
		public SqlDom Function { get; }
		public ImmutableArray<SqlDom> Args { get; }

		public CallSqlDom(SqlDom function, ImmutableArray<SqlDom> args) =>
			 (Function, Args) = (function, args);

		public override void WriteToStream(IndentStream stream)
		{
			Function.WriteToStream(stream);
			stream.Write("(");
			foreach (var item in Args.Select((value, idx) => new { value, idx }))
			{
				if (item.idx != 0)
				{
					stream.Write(", ");
				}
				item.value.WriteToStream(stream);
			}
			stream.Write(")");
		}
	}
}
