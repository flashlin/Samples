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