namespace PidginDemo.LinqExpressions;

public class IntegerExpr : LinqExpr
{
    public int Value { get; }

    public IntegerExpr(int value)
    {
        Value = value;
    }

    public override bool Equals(LinqExpr? other)
        => other is IntegerExpr x && Value == x.Value;

    public override int GetHashCode() => Value;
}

public enum UnaryOperatorType
{
    Neg,
    Complement
}

public class UnaryOpExpr : LinqExpr
{
    public UnaryOperatorType Type { get; }

    public LinqExpr Expr { get; }

    public UnaryOpExpr(UnaryOperatorType type, LinqExpr expr)
    {
        Type = type;
        Expr = expr;
    }

    public override bool Equals(LinqExpr? other)
        => other is UnaryOpExpr u
           && Type == u.Type
           && Expr.Equals(u.Expr);

    public override int GetHashCode() => HashCode.Combine(Type, Expr);
}