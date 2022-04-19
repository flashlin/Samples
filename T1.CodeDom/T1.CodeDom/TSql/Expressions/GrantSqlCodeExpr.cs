using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class GrantSqlCodeExpr : SqlCodeExpr
	{
		public List<string> PermissionList { get; set; }
		public SqlCodeExpr OnObjectId { get; set; }
		public List<SqlCodeExpr> TargetList { get; set; }
		public SqlCodeExpr AsDbo { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("GRANT ");
			PermissionList.Select(x => x.ToUpper()).WriteToStreamWithComma(stream);

			if( OnObjectId != null)
			{
				stream.WriteLine();
				stream.Write("ON ");
				OnObjectId.WriteToStream(stream);
			}

			stream.Write(" TO ");
			TargetList.WriteToStreamWithComma(stream);

			if( AsDbo != null)
			{
				stream.Write(" AS ");
				AsDbo.WriteToStream(stream);
			}
		}
	}
}