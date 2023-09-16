public abstract class SqlExpr : IEquatable<SqlExpr>
{
    public abstract bool Equals(SqlExpr? other);
    public override bool Equals(object? obj) => Equals(obj as SqlExpr);
    public abstract override int GetHashCode();
}