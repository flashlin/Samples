using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateRoleSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE ROLE ");
            RoleName.WriteToStream(stream);
            stream.Write(" AUTHORIZATION ");
            SchemaName.WriteToStream(stream);
        }

        public SqlCodeExpr RoleName { get; set; }
        public SqlCodeExpr SchemaName { get; set; }
    }
}