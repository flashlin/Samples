namespace T1.SqlSharp.Expressions;

public class SqlFuncTableSource : SqlTableSource
{
    public new SqlType SqlType { get; } = SqlType.FuncTableSource;
    public required SqlFunctionExpression Function { get; set; }
}