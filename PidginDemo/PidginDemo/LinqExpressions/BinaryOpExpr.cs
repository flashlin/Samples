namespace PidginDemo.LinqExpressions;

public class BinaryOpExpr : LinqExpr
{
    public BinaryOperatorType Type { get; }

    public LinqExpr Left { get; }

    public LinqExpr Right { get; }

    public BinaryOpExpr(BinaryOperatorType type, LinqExpr left, LinqExpr right)
    {
        Type = type;
        Left = left;
        Right = right;
    }

    public override bool Equals(LinqExpr? other)
        => other is BinaryOpExpr b
           && Type == b.Type
           && Left.Equals(b.Left)
           && Right.Equals(b.Right);

    public override int GetHashCode() => HashCode.Combine(Type, Left, Right);
}