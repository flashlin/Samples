using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class VariableSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"{Name}");	
        }

        public string Name { get; set; }
    }
}