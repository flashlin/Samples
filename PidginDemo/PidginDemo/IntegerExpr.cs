public class IntegerExpr : SqlExpr
{
    public int Value { get; }

    public IntegerExpr(int value)
    {
        Value = value;
    }

    public override bool Equals(SqlExpr? other)
        => other is IntegerExpr x && Value == x.Value;

    public override int GetHashCode() => Value;
}

public enum UnaryOperatorType
{
    Neg,
    Complement
}

public class UnaryOpExpr : SqlExpr
{
    public UnaryOperatorType Type { get; }

    public SqlExpr Expr { get; }

    public UnaryOpExpr(UnaryOperatorType type, SqlExpr expr)
    {
        Type = type;
        Expr = expr;
    }

    public override bool Equals(SqlExpr? other)
        => other is UnaryOpExpr u
           && Type == u.Type
           && Expr.Equals(u.Expr);

    public override int GetHashCode() => HashCode.Combine(Type, Expr);
}