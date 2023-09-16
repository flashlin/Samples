public class IdentifierExpr : SqlExpr
{
    public string Name { get; }

    public IdentifierExpr(string name)
    {
        Name = name;
    }

    public override bool Equals(SqlExpr? other)
        => other is IdentifierExpr x && Name == x.Name;

    public override int GetHashCode() => Name.GetHashCode(StringComparison.Ordinal);
}