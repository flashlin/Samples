using PreviewLibrary.PrattParsers.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.PrattParsers
{
	public class UpdateSqlDom : SqlDom
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("UPDATE");
		}
	}
}