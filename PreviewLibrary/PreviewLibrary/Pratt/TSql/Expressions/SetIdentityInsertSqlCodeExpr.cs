using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class SetIdentityInsertSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr ObjectId { get; set; }
		public string Toggle { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SET IDENTITY_INSERT ");
			ObjectId.WriteToStream(stream);
			stream.Write($" {Toggle.ToUpper()}");
		}
	}
}