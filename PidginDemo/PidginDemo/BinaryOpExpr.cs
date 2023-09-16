public class BinaryOpExpr : SqlExpr
{
    public BinaryOperatorType Type { get; }

    public SqlExpr Left { get; }

    public SqlExpr Right { get; }

    public BinaryOpExpr(BinaryOperatorType type, SqlExpr left, SqlExpr right)
    {
        Type = type;
        Left = left;
        Right = right;
    }

    public override bool Equals(SqlExpr? other)
        => other is BinaryOpExpr b
           && Type == b.Type
           && Left.Equals(b.Left)
           && Right.Equals(b.Right);

    public override int GetHashCode() => HashCode.Combine(Type, Left, Right);
}