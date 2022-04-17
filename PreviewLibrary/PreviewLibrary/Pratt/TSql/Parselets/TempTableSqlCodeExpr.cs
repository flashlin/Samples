using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class TempTableSqlCodeExpr : SqlCodeExpr 
	{
		public string Name { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Name);
		}
	}
}