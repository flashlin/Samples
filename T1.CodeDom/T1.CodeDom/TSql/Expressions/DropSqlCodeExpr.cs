using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
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