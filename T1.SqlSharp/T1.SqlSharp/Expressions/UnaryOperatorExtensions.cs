namespace T1.SqlSharp.Expressions;

public static class UnaryOperatorExtensions
{
    public static string ToSql(this UnaryOperator op)
    {
        return op switch
        {
            UnaryOperator.BitwiseNot => "~",
            UnaryOperator.Not => "NOT",
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, null)
        };
    }
}