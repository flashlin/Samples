using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers.Expressions
{
	public class AliasSqlDom : SqlDom
	{
		public SqlDom Left { get; set; }
		public SqlDom AliasName { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Left.WriteToStream(stream);
			stream.Write(" AS ");
			AliasName.WriteToStream(stream);
		}
	}
}