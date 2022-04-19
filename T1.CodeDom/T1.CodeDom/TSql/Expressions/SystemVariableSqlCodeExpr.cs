using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class SystemVariableSqlCodeExpr : SqlCodeExpr
	{
		public string Name { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Name);
		}
	}
}