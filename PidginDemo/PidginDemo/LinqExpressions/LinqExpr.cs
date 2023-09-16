namespace PidginDemo.LinqExpressions;

public abstract class LinqExpr : IEquatable<LinqExpr>
{
    public abstract bool Equals(LinqExpr? other);
    public override bool Equals(object? obj) => Equals(obj as LinqExpr);
    public abstract override int GetHashCode();
}