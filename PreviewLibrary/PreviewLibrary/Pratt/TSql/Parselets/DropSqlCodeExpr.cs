using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class DropSqlCodeExpr : SqlCodeExpr
	{
		public string TargetId { get; set; }
		public SqlCodeExpr ObjectId { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"DROP {TargetId}");

			stream.Write(" ");
			ObjectId.WriteToStream(stream);
		}
	}
}