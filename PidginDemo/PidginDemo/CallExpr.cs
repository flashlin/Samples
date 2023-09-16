using System.Collections.Immutable;

public class CallExpr : SqlExpr
{
    public SqlExpr Expr { get; }

    public ImmutableArray<SqlExpr> Arguments { get; }

    public CallExpr(SqlExpr expr, ImmutableArray<SqlExpr> arguments)
    {
        Expr = expr;
        Arguments = arguments;
    }

    public override bool Equals(SqlExpr? other)
        => other is CallExpr x
           && Expr.Equals(x.Expr)
           && Arguments.SequenceEqual(x.Arguments);

    public override int GetHashCode() => HashCode.Combine(Expr, Arguments);
}