namespace T1.SqlSharp.Expressions;

public class SqlCaseExpression : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Case;
    public required ISqlExpression InputExpr { get; set; }
    public List<SqlCaseWhenExpression> Whens { get; set; } = [];

    public string ToSql()
    {
        throw new NotImplementedException();
    }
}