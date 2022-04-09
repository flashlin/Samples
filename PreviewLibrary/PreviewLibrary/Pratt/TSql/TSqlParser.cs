using PreviewLibrary.Pratt.Core;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlParser : PrattParser<SqlCodeDom>
	{
		public TSqlParser(IScanner scanner) : base(scanner)
		{
		}
	}
}
