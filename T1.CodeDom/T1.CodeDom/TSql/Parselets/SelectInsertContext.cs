using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
    public class SelectInsertContext
    {
        public TopSqlCodeExpr TopCount { get; set; }
        public List<SqlCodeExpr> Columns { get; set; }
        public SqlCodeExpr IntoTable { get; set; }
    }
}