namespace T1.SqlSharp.Expressions;

public class SqlLabelStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.LabelStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_LabelStatement(this);
    }

    public string Label { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{Label}:";
    }
}
