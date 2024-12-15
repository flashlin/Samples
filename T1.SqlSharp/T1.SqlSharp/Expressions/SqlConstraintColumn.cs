namespace T1.SqlSharp.Expressions;

public class SqlConstraintColumn : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.Constraint;
    public TextSpan Span { get; set; } = new();
    public string ColumnName { get; set; } = string.Empty;
    public string Order { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{ColumnName} {Order}";
    }
}