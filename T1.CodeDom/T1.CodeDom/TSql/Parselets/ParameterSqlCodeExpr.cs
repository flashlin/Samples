using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class ParameterSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public bool IsOutput { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			if (IsOutput)
			{
				stream.Write(" OUT");
			}
		}
	}
}