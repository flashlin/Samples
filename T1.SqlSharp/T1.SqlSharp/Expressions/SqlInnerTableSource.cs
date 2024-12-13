namespace T1.SqlSharp.Expressions;

public class SqlInnerTableSource : SqlTableSource
{
    public new SqlType SqlType { get; } = SqlType.InnerTableSource;
    public required ISqlExpression Inner { get; set; }
}