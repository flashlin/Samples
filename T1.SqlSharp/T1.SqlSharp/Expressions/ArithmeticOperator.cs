namespace T1.SqlSharp.Expressions;

public enum ArithmeticOperator
{
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

public static class ArithmeticOperatorExtensions
{
    public static string ToSql(this ArithmeticOperator op)
    {
        return op switch
        {
            ArithmeticOperator.Add => "+",
            ArithmeticOperator.Subtract => "-",
            ArithmeticOperator.Multiply => "*",
            ArithmeticOperator.Divide => "/",
            ArithmeticOperator.Modulo => "%",
            ArithmeticOperator.BitwiseAnd => "&",
            ArithmeticOperator.BitwiseOr => "|",
            ArithmeticOperator.BitwiseXor => "^",
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, null),
        };
    }
    
    public static ArithmeticOperator ToArithmeticOperator(this string op)
    {
        return op switch
        {
            "+" => ArithmeticOperator.Add,
            "-" => ArithmeticOperator.Subtract,
            "*" => ArithmeticOperator.Multiply,
            "/" => ArithmeticOperator.Divide,
            "%" => ArithmeticOperator.Modulo,
            "&" => ArithmeticOperator.BitwiseAnd,
            "|" => ArithmeticOperator.BitwiseOr,
            "^" => ArithmeticOperator.BitwiseXor,
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, null),
        };
    }
}