using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class AliasSqlDom : SqlDom
	{
		public SqlDom Left { get; set; }
		public SqlDom Name { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(" AS ");
			Name.WriteToStream(stream);
		}
	}
}