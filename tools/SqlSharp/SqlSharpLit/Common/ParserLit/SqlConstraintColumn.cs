using SqlSharpLit.Common.ParserLit.Expressions;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlConstraintColumn : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.Constraint;
    public string ColumnName { get; set; } = string.Empty;
    public string Order { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{ColumnName} {Order}";
    }
}