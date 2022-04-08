using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class CreateProcedureSqlDom : SqlDom
	{
		public SqlDom Name { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE PROCEDURE ");
			Name.WriteToStream(stream);
		}
	}
}