namespace T1.SqlSharp.Expressions;

public enum ArithmeticOperator
{
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    ModuloAssign,
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
            ArithmeticOperator.AddAssign => "+=",
            ArithmeticOperator.SubtractAssign => "-=",
            ArithmeticOperator.MultiplyAssign => "*=",
            ArithmeticOperator.DivideAssign => "/=",
            ArithmeticOperator.ModuloAssign => "%=",
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
            "+=" => ArithmeticOperator.AddAssign,
            "-=" => ArithmeticOperator.SubtractAssign,
            "*=" => ArithmeticOperator.MultiplyAssign,
            "/=" => ArithmeticOperator.DivideAssign,
            "%=" => ArithmeticOperator.ModuloAssign,
            "&" => ArithmeticOperator.BitwiseAnd,
            "|" => ArithmeticOperator.BitwiseOr,
            "^" => ArithmeticOperator.BitwiseXor,
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, null),
        };
    }
}