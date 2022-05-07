using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateLoginSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE LOGIN ");
            LoginName.WriteToStream(stream);
            stream.Write(" WITH PASSWORD = ");
            Password.WriteToStream(stream);
        }

        public SqlCodeExpr LoginName { get; set; }
        public SqlCodeExpr Password { get; set; }
    }
}